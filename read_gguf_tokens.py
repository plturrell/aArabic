#!/usr/bin/env python3
import struct
import sys

# GGUF type sizes
TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 8, 8: 8}

def read_value(f, vtype):
    """Read a value and return it"""
    if vtype == 0:  # uint8
        return struct.unpack('<B', f.read(1))[0]
    elif vtype == 1:  # int8
        return struct.unpack('<b', f.read(1))[0]
    elif vtype == 2:  # uint16
        return struct.unpack('<H', f.read(2))[0]
    elif vtype == 3:  # int16
        return struct.unpack('<h', f.read(2))[0]
    elif vtype == 4:  # uint32
        return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5:  # int32
        return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6:  # float32
        return struct.unpack('<f', f.read(4))[0]
    elif vtype == 7:  # bool
        return struct.unpack('<?', f.read(1))[0]
    elif vtype == 8:  # string
        str_len = struct.unpack('<Q', f.read(8))[0]
        return f.read(str_len).decode('utf-8', errors='replace')
    elif vtype == 9:  # array
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        # For large arrays, just skip
        if arr_len > 100:
            for _ in range(arr_len):
                read_value(f, arr_type)
            return f"array[{arr_len}]"
        return [read_value(f, arr_type) for _ in range(arr_len)]
    elif vtype == 10:  # uint64
        return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11:  # int64
        return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12:  # float64
        return struct.unpack('<d', f.read(8))[0]
    return None

def read_gguf_metadata(path, show_tokens=None):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            print(f'Not a GGUF file: {magic}')
            return

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]

        print(f'GGUF v{version}, {tensor_count} tensors, {metadata_count} metadata keys\n')

        for i in range(int(metadata_count)):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8', errors='replace')
            value_type = struct.unpack('<I', f.read(4))[0]

            # Special handling for tokens array
            if key == 'tokenizer.ggml.tokens' and show_tokens:
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                print(f'Vocab size: {arr_len}')
                for idx in range(arr_len):
                    token = read_value(f, arr_type)
                    if idx in show_tokens:
                        print(f'  Token {idx}: {repr(token)}')
                continue

            value = read_value(f, value_type)

            # Show token-related keys
            if 'token' in key.lower() or 'eos' in key.lower() or 'bos' in key.lower() or 'pad' in key.lower() or 'vocab' in key.lower():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + '...'
                print(f'{key} = {value}')

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/Users/user/Documents/arabic_folder/vendor/layerModels/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_0.gguf'
    # Show tokens 0-10, 221, and some special ones
    show_tokens = set(range(11)) | {221, 222, 223, 7, 1, 0}
    read_gguf_metadata(path, show_tokens=show_tokens)

