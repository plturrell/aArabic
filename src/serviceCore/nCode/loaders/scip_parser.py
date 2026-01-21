"""
SCIP Protobuf Parser for Python

Parses SCIP index files and extracts symbols, occurrences, and relationships.
"""

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class Relationship:
    """Symbol relationship"""
    symbol: str = ""
    is_reference: bool = False
    is_implementation: bool = False
    is_type_definition: bool = False
    is_definition: bool = False


@dataclass
class SymbolInfo:
    """Symbol information with documentation and relationships"""
    symbol: str = ""
    documentation: List[str] = field(default_factory=list)
    kind: int = 0
    display_name: str = ""
    enclosing_symbol: str = ""
    relationships: List[Relationship] = field(default_factory=list)


@dataclass
class Occurrence:
    """Symbol occurrence in source code"""
    start_line: int = 0
    start_char: int = 0
    end_line: int = 0
    end_char: int = 0
    symbol: str = ""
    symbol_roles: int = 0
    syntax_kind: int = 0
    
    def is_definition(self) -> bool:
        return (self.symbol_roles & 0x1) != 0


@dataclass
class Document:
    """Source file document"""
    language: str = ""
    relative_path: str = ""
    text: str = ""
    occurrences: List[Occurrence] = field(default_factory=list)
    symbols: List[SymbolInfo] = field(default_factory=list)


@dataclass
class ToolInfo:
    """Indexer tool information"""
    name: str = ""
    version: str = ""
    arguments: List[str] = field(default_factory=list)


@dataclass
class Metadata:
    """Index metadata"""
    version: int = 0
    tool_info: ToolInfo = field(default_factory=ToolInfo)
    project_root: str = ""
    text_encoding: int = 0


@dataclass
class ScipIndex:
    """Complete SCIP index"""
    metadata: Metadata = field(default_factory=Metadata)
    documents: List[Document] = field(default_factory=list)
    external_symbols: List[SymbolInfo] = field(default_factory=list)


def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Read unsigned varint, returns (value, bytes_consumed)"""
    result = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if (byte & 0x80) == 0:
            return result, pos
        shift += 7
    raise ValueError("Unexpected end of data")


def read_length_delimited(data: bytes, pos: int) -> tuple[bytes, int]:
    """Read length-delimited field, returns (data, new_pos)"""
    length, pos = read_varint(data, pos)
    return data[pos:pos + length], pos + length


def read_tag(data: bytes, pos: int) -> tuple[int, int, int]:
    """Read protobuf tag, returns (field_number, wire_type, new_pos)"""
    tag, pos = read_varint(data, pos)
    return tag >> 3, tag & 0x7, pos


def skip_field(data: bytes, pos: int, wire_type: int) -> int:
    """Skip unknown field based on wire type"""
    if wire_type == 0:  # Varint
        _, pos = read_varint(data, pos)
    elif wire_type == 1:  # Fixed64
        pos += 8
    elif wire_type == 2:  # Length-delimited
        length, pos = read_varint(data, pos)
        pos += length
    elif wire_type == 5:  # Fixed32
        pos += 4
    return pos


def read_packed_int32(data: bytes) -> List[int]:
    """Read packed repeated int32"""
    result = []
    pos = 0
    while pos < len(data):
        val, pos = read_varint(data, pos)
        result.append(val)
    return result


def parse_relationship(data: bytes) -> Relationship:
    """Parse Relationship message"""
    rel = Relationship()
    pos = 0
    while pos < len(data):
        field_num, wire_type, pos = read_tag(data, pos)
        if field_num == 1:  # symbol
            rel.symbol, pos = read_length_delimited(data, pos)
            rel.symbol = rel.symbol.decode('utf-8')
        elif field_num == 2:  # is_reference
            val, pos = read_varint(data, pos)
            rel.is_reference = val != 0
        elif field_num == 3:  # is_implementation
            val, pos = read_varint(data, pos)
            rel.is_implementation = val != 0
        elif field_num == 4:  # is_type_definition
            val, pos = read_varint(data, pos)
            rel.is_type_definition = val != 0
        elif field_num == 5:  # is_definition
            val, pos = read_varint(data, pos)
            rel.is_definition = val != 0
        else:
            pos = skip_field(data, pos, wire_type)
    return rel


def parse_symbol_info(data: bytes) -> SymbolInfo:
    """Parse SymbolInformation message"""
    info = SymbolInfo()
    pos = 0
    while pos < len(data):
        field_num, wire_type, pos = read_tag(data, pos)
        if field_num == 1:  # symbol
            info.symbol, pos = read_length_delimited(data, pos)
            info.symbol = info.symbol.decode('utf-8')
        elif field_num == 3:  # documentation
            doc, pos = read_length_delimited(data, pos)
            info.documentation.append(doc.decode('utf-8'))
        elif field_num == 4:  # relationships
            rel_data, pos = read_length_delimited(data, pos)
            info.relationships.append(parse_relationship(rel_data))
        elif field_num == 5:  # kind
            info.kind, pos = read_varint(data, pos)
        elif field_num == 6:  # display_name
            info.display_name, pos = read_length_delimited(data, pos)
            info.display_name = info.display_name.decode('utf-8')
        elif field_num == 8:  # enclosing_symbol
            info.enclosing_symbol, pos = read_length_delimited(data, pos)
            info.enclosing_symbol = info.enclosing_symbol.decode('utf-8')
        else:
            pos = skip_field(data, pos, wire_type)
    return info


def parse_occurrence(data: bytes) -> Occurrence:
    """Parse Occurrence message"""
    occ = Occurrence()
    pos = 0
    while pos < len(data):
        field_num, wire_type, pos = read_tag(data, pos)
        if field_num == 1:  # range
            range_data, pos = read_length_delimited(data, pos)
            ranges = read_packed_int32(range_data)
            if len(ranges) >= 3:
                occ.start_line = ranges[0]
                occ.start_char = ranges[1]
                if len(ranges) == 3:
                    occ.end_line = occ.start_line
                    occ.end_char = ranges[2]
                else:
                    occ.end_line = ranges[2]
                    occ.end_char = ranges[3]
        elif field_num == 2:  # symbol
            occ.symbol, pos = read_length_delimited(data, pos)
            occ.symbol = occ.symbol.decode('utf-8')
        elif field_num == 3:  # symbol_roles
            occ.symbol_roles, pos = read_varint(data, pos)
        elif field_num == 5:  # syntax_kind
            occ.syntax_kind, pos = read_varint(data, pos)
        else:
            pos = skip_field(data, pos, wire_type)
    return occ


def parse_document(data: bytes) -> Document:
    """Parse Document message"""
    doc = Document()
    pos = 0
    while pos < len(data):
        field_num, wire_type, pos = read_tag(data, pos)
        if field_num == 1:  # relative_path
            doc.relative_path, pos = read_length_delimited(data, pos)
            doc.relative_path = doc.relative_path.decode('utf-8')
        elif field_num == 2:  # occurrences
            occ_data, pos = read_length_delimited(data, pos)
            doc.occurrences.append(parse_occurrence(occ_data))
        elif field_num == 3:  # symbols
            sym_data, pos = read_length_delimited(data, pos)
            doc.symbols.append(parse_symbol_info(sym_data))
        elif field_num == 4:  # language
            doc.language, pos = read_length_delimited(data, pos)
            doc.language = doc.language.decode('utf-8')
        elif field_num == 5:  # text
            doc.text, pos = read_length_delimited(data, pos)
            doc.text = doc.text.decode('utf-8')
        else:
            pos = skip_field(data, pos, wire_type)
    return doc


def parse_tool_info(data: bytes) -> ToolInfo:
    """Parse ToolInfo message"""
    info = ToolInfo()
    pos = 0
    while pos < len(data):
        field_num, wire_type, pos = read_tag(data, pos)
        if field_num == 1:  # name
            info.name, pos = read_length_delimited(data, pos)
            info.name = info.name.decode('utf-8')
        elif field_num == 2:  # version
            info.version, pos = read_length_delimited(data, pos)
            info.version = info.version.decode('utf-8')
        elif field_num == 3:  # arguments
            arg, pos = read_length_delimited(data, pos)
            info.arguments.append(arg.decode('utf-8'))
        else:
            pos = skip_field(data, pos, wire_type)
    return info


def parse_metadata(data: bytes) -> Metadata:
    """Parse Metadata message"""
    meta = Metadata()
    pos = 0
    while pos < len(data):
        field_num, wire_type, pos = read_tag(data, pos)
        if field_num == 1:  # version
            meta.version, pos = read_varint(data, pos)
        elif field_num == 2:  # tool_info
            tool_data, pos = read_length_delimited(data, pos)
            meta.tool_info = parse_tool_info(tool_data)
        elif field_num == 3:  # project_root
            meta.project_root, pos = read_length_delimited(data, pos)
            meta.project_root = meta.project_root.decode('utf-8')
        elif field_num == 4:  # text_encoding
            meta.text_encoding, pos = read_varint(data, pos)
        else:
            pos = skip_field(data, pos, wire_type)
    return meta


def parse_index(data: bytes) -> ScipIndex:
    """Parse complete SCIP Index"""
    index = ScipIndex()
    pos = 0
    while pos < len(data):
        field_num, wire_type, pos = read_tag(data, pos)
        if field_num == 1:  # metadata
            meta_data, pos = read_length_delimited(data, pos)
            index.metadata = parse_metadata(meta_data)
        elif field_num == 2:  # documents
            doc_data, pos = read_length_delimited(data, pos)
            index.documents.append(parse_document(doc_data))
        elif field_num == 3:  # external_symbols
            sym_data, pos = read_length_delimited(data, pos)
            index.external_symbols.append(parse_symbol_info(sym_data))
        else:
            pos = skip_field(data, pos, wire_type)
    return index


def load_scip_file(path: str) -> ScipIndex:
    """Load and parse a SCIP index file"""
    with open(path, 'rb') as f:
        data = f.read()
    return parse_index(data)

