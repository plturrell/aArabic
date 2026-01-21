// OOXML Parser Tests
const std = @import("std");
const testing = std.testing;
const ooxml = @import("../parsers/ooxml.zig");

test "OOXML Package initialization" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    try testing.expect(package.content_types.defaults.count() == 0);
    try testing.expect(package.content_types.overrides.count() == 0);
    try testing.expect(package.relationships.count() == 0);
}

test "Content types - defaults and overrides" {
    const allocator = testing.allocator;
    
    const xml_data =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
        \\  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
        \\  <Default Extension="xml" ContentType="application/xml"/>
        \\  <Default Extension="png" ContentType="image/png"/>
        \\  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
        \\  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
        \\</Types>
    ;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    try package.parseContentTypes(xml_data);
    
    // Verify defaults
    try testing.expectEqual(@as(usize, 3), package.content_types.defaults.count());
    
    // Verify overrides
    try testing.expectEqual(@as(usize, 2), package.content_types.overrides.count());
    
    // Test content type lookup by override
    const doc_type = package.getContentType("/word/document.xml");
    try testing.expect(doc_type != null);
    try testing.expect(std.mem.eql(u8, doc_type.?, ooxml.ContentTypeValues.word_document));
    
    // Test content type lookup by extension
    const xml_type = package.getContentType("test.xml");
    try testing.expect(xml_type != null);
    try testing.expect(std.mem.eql(u8, xml_type.?, "application/xml"));
}

test "Relationships parsing with internal and external targets" {
    const allocator = testing.allocator;
    
    const rels_xml =
        \\<?xml version="1.0"?>
        \\<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
        \\  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
        \\  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com" TargetMode="External"/>
        \\  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
        \\</Relationships>
    ;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    try package.parseRelationships("", rels_xml);
    
    const rels = package.getRelationships("");
    try testing.expect(rels != null);
    try testing.expectEqual(@as(usize, 3), rels.?.items.len);
    
    // Check first relationship (internal)
    const rel1 = rels.?.items[0];
    try testing.expect(std.mem.eql(u8, rel1.id, "rId1"));
    try testing.expect(std.mem.eql(u8, rel1.type, ooxml.RelationshipTypes.office_document));
    try testing.expect(rel1.target_mode == .internal);
    
    // Check second relationship (external)
    const rel2 = rels.?.items[1];
    try testing.expect(std.mem.eql(u8, rel2.id, "rId2"));
    try testing.expect(rel2.target_mode == .external);
    try testing.expect(std.mem.eql(u8, rel2.target, "https://example.com"));
}

test "Extract source part name from relationship paths" {
    const allocator = testing.allocator;
    
    // Test word document relationship
    const result1 = try ooxml.extractSourcePartName(allocator, "word/_rels/document.xml.rels");
    defer allocator.free(result1);
    try testing.expect(std.mem.eql(u8, result1, "word/document.xml"));
    
    // Test root relationship
    const result2 = try ooxml.extractSourcePartName(allocator, "_rels/.rels");
    defer allocator.free(result2);
    try testing.expect(std.mem.eql(u8, result2, ""));
    
    // Test nested path
    const result3 = try ooxml.extractSourcePartName(allocator, "xl/worksheets/_rels/sheet1.xml.rels");
    defer allocator.free(result3);
    try testing.expect(std.mem.eql(u8, result3, "xl/worksheets/sheet1.xml"));
}

test "Resolve relationship targets" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Test absolute path
    const abs_result = try package.resolveTarget("word/document.xml", "/word/styles.xml");
    defer allocator.free(abs_result);
    try testing.expect(std.mem.eql(u8, abs_result, "word/styles.xml"));
    
    // Test relative path from same directory
    const rel_result = try package.resolveTarget("word/document.xml", "styles.xml");
    defer allocator.free(rel_result);
    try testing.expect(std.mem.eql(u8, rel_result, "word/styles.xml"));
    
    // Test relative path with subdirectory
    const subdir_result = try package.resolveTarget("word/document.xml", "media/image1.png");
    defer allocator.free(subdir_result);
    try testing.expect(std.mem.eql(u8, subdir_result, "word/media/image1.png"));
}

test "Package validation - valid package" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add minimal valid content types
    try package.content_types.defaults.put(
        try allocator.dupe(u8, "xml"),
        try allocator.dupe(u8, "application/xml")
    );
    
    // Add root relationships with office document
    var root_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try root_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.office_document),
        .target = try allocator.dupe(u8, "word/document.xml"),
        .target_mode = .internal,
    });
    
    try package.relationships.put(try allocator.dupe(u8, ""), root_rels);
    
    // Should validate successfully
    try ooxml.validatePackage(&package);
}

test "Package validation - missing content types" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add relationships but no content types
    var root_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try root_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.office_document),
        .target = try allocator.dupe(u8, "word/document.xml"),
        .target_mode = .internal,
    });
    
    try package.relationships.put(try allocator.dupe(u8, ""), root_rels);
    
    // Should fail validation
    try testing.expectError(error.MissingContentTypes, ooxml.validatePackage(&package));
}

test "Package validation - missing root relationships" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add content types but no relationships
    try package.content_types.defaults.put(
        try allocator.dupe(u8, "xml"),
        try allocator.dupe(u8, "application/xml")
    );
    
    // Should fail validation
    try testing.expectError(error.MissingRootRelationships, ooxml.validatePackage(&package));
}

test "Package validation - missing office document relationship" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add content types
    try package.content_types.defaults.put(
        try allocator.dupe(u8, "xml"),
        try allocator.dupe(u8, "application/xml")
    );
    
    // Add root relationships but wrong type
    var root_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try root_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.extended_properties),
        .target = try allocator.dupe(u8, "docProps/app.xml"),
        .target_mode = .internal,
    });
    
    try package.relationships.put(try allocator.dupe(u8, ""), root_rels);
    
    // Should fail validation
    try testing.expectError(error.MissingOfficeDocumentRelationship, ooxml.validatePackage(&package));
}

test "Multiple relationship files" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add root relationships
    const root_rels_xml =
        \\<?xml version="1.0"?>
        \\<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
        \\  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
        \\</Relationships>
    ;
    try package.parseRelationships("", root_rels_xml);
    
    // Add document relationships
    const doc_rels_xml =
        \\<?xml version="1.0"?>
        \\<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
        \\  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
        \\  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
        \\</Relationships>
    ;
    try package.parseRelationships("word/document.xml", doc_rels_xml);
    
    // Verify both relationship sets exist
    const root_rels = package.getRelationships("");
    try testing.expect(root_rels != null);
    try testing.expectEqual(@as(usize, 1), root_rels.?.items.len);
    
    const doc_rels = package.getRelationships("word/document.xml");
    try testing.expect(doc_rels != null);
    try testing.expectEqual(@as(usize, 2), doc_rels.?.items.len);
}

test "Part content type resolution priority" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add default for xml extension
    try package.content_types.defaults.put(
        try allocator.dupe(u8, "xml"),
        try allocator.dupe(u8, "application/xml")
    );
    
    // Add override for specific part
    try package.content_types.overrides.put(
        try allocator.dupe(u8, "/word/document.xml"),
        try allocator.dupe(u8, ooxml.ContentTypeValues.word_document)
    );
    
    // Override should take precedence
    const doc_type = package.getContentType("/word/document.xml");
    try testing.expect(doc_type != null);
    try testing.expect(std.mem.eql(u8, doc_type.?, ooxml.ContentTypeValues.word_document));
    
    // Other xml files should use default
    const other_type = package.getContentType("/other.xml");
    try testing.expect(other_type != null);
    try testing.expect(std.mem.eql(u8, other_type.?, "application/xml"));
}

test "Relationship type constants" {
    // Verify all relationship type constants are defined correctly
    try testing.expect(std.mem.indexOf(u8, ooxml.RelationshipTypes.office_document, "officeDocument") != null);
    try testing.expect(std.mem.indexOf(u8, ooxml.RelationshipTypes.styles, "styles") != null);
    try testing.expect(std.mem.indexOf(u8, ooxml.RelationshipTypes.image, "image") != null);
    try testing.expect(std.mem.indexOf(u8, ooxml.RelationshipTypes.worksheet, "worksheet") != null);
    try testing.expect(std.mem.indexOf(u8, ooxml.RelationshipTypes.slide, "slide") != null);
}

test "Content type constants" {
    // Verify all content type constants are defined correctly
    try testing.expect(std.mem.indexOf(u8, ooxml.ContentTypeValues.word_document, "wordprocessingml") != null);
    try testing.expect(std.mem.indexOf(u8, ooxml.ContentTypeValues.excel_workbook, "spreadsheetml") != null);
    try testing.expect(std.mem.indexOf(u8, ooxml.ContentTypeValues.powerpoint_presentation, "presentationml") != null);
}

// ===== DAY 17: Enhanced OOXML Tests =====

test "Find main document part" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add root relationships
    var root_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try root_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.office_document),
        .target = try allocator.dupe(u8, "word/document.xml"),
        .target_mode = .internal,
    });
    try package.relationships.put(try allocator.dupe(u8, ""), root_rels);
    
    // Find main document
    const main_doc = try ooxml.findMainDocument(&package);
    try testing.expect(std.mem.eql(u8, main_doc, "word/document.xml"));
}

test "Find parts by type" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add document relationships with images
    var doc_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try doc_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.image),
        .target = try allocator.dupe(u8, "media/image1.png"),
        .target_mode = .internal,
    });
    try doc_rels.append(.{
        .id = try allocator.dupe(u8, "rId2"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.image),
        .target = try allocator.dupe(u8, "media/image2.jpg"),
        .target_mode = .internal,
    });
    try doc_rels.append(.{
        .id = try allocator.dupe(u8, "rId3"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.styles),
        .target = try allocator.dupe(u8, "styles.xml"),
        .target_mode = .internal,
    });
    try package.relationships.put(try allocator.dupe(u8, "word/document.xml"), doc_rels);
    
    // Find all images
    var images = try ooxml.findPartsByType(&package, allocator, ooxml.RelationshipTypes.image);
    defer {
        for (images.items) |img| allocator.free(img);
        images.deinit();
    }
    
    try testing.expectEqual(@as(usize, 2), images.items.len);
}

test "Get package type - DOCX" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add DOCX main document
    var root_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try root_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.office_document),
        .target = try allocator.dupe(u8, "word/document.xml"),
        .target_mode = .internal,
    });
    try package.relationships.put(try allocator.dupe(u8, ""), root_rels);
    
    // Add content type for word document
    try package.content_types.overrides.put(
        try allocator.dupe(u8, "word/document.xml"),
        try allocator.dupe(u8, ooxml.ContentTypeValues.word_document)
    );
    
    const pkg_type = ooxml.getPackageType(&package);
    try testing.expectEqual(ooxml.PackageType.docx, pkg_type);
}

test "Get package type - XLSX" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add XLSX main document
    var root_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try root_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.office_document),
        .target = try allocator.dupe(u8, "xl/workbook.xml"),
        .target_mode = .internal,
    });
    try package.relationships.put(try allocator.dupe(u8, ""), root_rels);
    
    // Add content type for excel workbook
    try package.content_types.overrides.put(
        try allocator.dupe(u8, "xl/workbook.xml"),
        try allocator.dupe(u8, ooxml.ContentTypeValues.excel_workbook)
    );
    
    const pkg_type = ooxml.getPackageType(&package);
    try testing.expectEqual(ooxml.PackageType.xlsx, pkg_type);
}

test "Get package type - PPTX" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add PPTX main document
    var root_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try root_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.office_document),
        .target = try allocator.dupe(u8, "ppt/presentation.xml"),
        .target_mode = .internal,
    });
    try package.relationships.put(try allocator.dupe(u8, ""), root_rels);
    
    // Add content type for powerpoint presentation
    try package.content_types.overrides.put(
        try allocator.dupe(u8, "ppt/presentation.xml"),
        try allocator.dupe(u8, ooxml.ContentTypeValues.powerpoint_presentation)
    );
    
    const pkg_type = ooxml.getPackageType(&package);
    try testing.expectEqual(ooxml.PackageType.pptx, pkg_type);
}

test "Digital signatures detection" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Package without signatures
    try testing.expect(!ooxml.hasDigitalSignatures(&package));
    
    // Add relationship with digital signature
    var root_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try root_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/package/2006/relationships/digital-signature/signature"),
        .target = try allocator.dupe(u8, "_xmlsignatures/sig1.xml"),
        .target_mode = .internal,
    });
    try package.relationships.put(try allocator.dupe(u8, ""), root_rels);
    
    // Should now detect signatures
    try testing.expect(ooxml.hasDigitalSignatures(&package));
}

test "Extract metadata from core properties" {
    const allocator = testing.allocator;
    
    const core_props_xml =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
        \\                   xmlns:dc="http://purl.org/dc/elements/1.1/"
        \\                   xmlns:dcterms="http://purl.org/dc/terms/">
        \\  <dc:title>Sample Document</dc:title>
        \\  <dc:creator>John Doe</dc:creator>
        \\  <dc:subject>OOXML Testing</dc:subject>
        \\  <dc:description>A test document for OOXML parser</dc:description>
        \\  <cp:keywords>test, ooxml, parser</cp:keywords>
        \\  <cp:lastModifiedBy>Jane Smith</cp:lastModifiedBy>
        \\  <dcterms:created>2026-01-17T10:00:00Z</dcterms:created>
        \\  <dcterms:modified>2026-01-17T12:00:00Z</dcterms:modified>
        \\</cp:coreProperties>
    ;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    var metadata = try ooxml.extractMetadata(&package, allocator, core_props_xml);
    defer metadata.deinit();
    
    // Verify metadata extraction
    try testing.expect(metadata.title != null);
    try testing.expect(std.mem.eql(u8, metadata.title.?, "Sample Document"));
    
    try testing.expect(metadata.creator != null);
    try testing.expect(std.mem.eql(u8, metadata.creator.?, "John Doe"));
    
    try testing.expect(metadata.subject != null);
    try testing.expect(std.mem.eql(u8, metadata.subject.?, "OOXML Testing"));
    
    try testing.expect(metadata.keywords != null);
    try testing.expect(std.mem.eql(u8, metadata.keywords.?, "test, ooxml, parser"));
}

test "Package statistics" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Add content types
    try package.content_types.defaults.put(
        try allocator.dupe(u8, "xml"),
        try allocator.dupe(u8, "application/xml")
    );
    try package.content_types.overrides.put(
        try allocator.dupe(u8, "/word/document.xml"),
        try allocator.dupe(u8, ooxml.ContentTypeValues.word_document)
    );
    
    // Add relationships with images and external refs
    var root_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try root_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.office_document),
        .target = try allocator.dupe(u8, "word/document.xml"),
        .target_mode = .internal,
    });
    try package.relationships.put(try allocator.dupe(u8, ""), root_rels);
    
    var doc_rels = std.ArrayList(ooxml.Relationship).init(allocator);
    try doc_rels.append(.{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.image),
        .target = try allocator.dupe(u8, "media/image1.png"),
        .target_mode = .internal,
    });
    try doc_rels.append(.{
        .id = try allocator.dupe(u8, "rId2"),
        .type = try allocator.dupe(u8, ooxml.RelationshipTypes.hyperlink),
        .target = try allocator.dupe(u8, "https://example.com"),
        .target_mode = .external,
    });
    try package.relationships.put(try allocator.dupe(u8, "word/document.xml"), doc_rels);
    
    const stats = ooxml.getPackageStats(&package);
    
    try testing.expectEqual(@as(usize, 3), stats.total_relationships);
    try testing.expectEqual(@as(usize, 2), stats.content_type_count);
    try testing.expect(stats.has_images);
    try testing.expect(stats.has_external_refs);
    try testing.expectEqual(ooxml.PackageType.docx, stats.package_type);
}

test "Metadata with missing fields" {
    const allocator = testing.allocator;
    
    const core_props_xml =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
        \\                   xmlns:dc="http://purl.org/dc/elements/1.1/">
        \\  <dc:title>Minimal Document</dc:title>
        \\</cp:coreProperties>
    ;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    var metadata = try ooxml.extractMetadata(&package, allocator, core_props_xml);
    defer metadata.deinit();
    
    // Verify only title is extracted
    try testing.expect(metadata.title != null);
    try testing.expect(std.mem.eql(u8, metadata.title.?, "Minimal Document"));
    
    // Other fields should be null
    try testing.expect(metadata.creator == null);
    try testing.expect(metadata.subject == null);
    try testing.expect(metadata.description == null);
}

test "Complex relationship resolution" {
    const allocator = testing.allocator;
    
    var package = ooxml.OOXMLPackage.init(allocator);
    defer package.deinit();
    
    // Test resolution from nested directory
    const result1 = try package.resolveTarget("word/document.xml", "../docProps/core.xml");
    defer allocator.free(result1);
    try testing.expect(std.mem.eql(u8, result1, "word/../docProps/core.xml"));
    
    // Test resolution with multiple path segments
    const result2 = try package.resolveTarget("xl/worksheets/sheet1.xml", "../sharedStrings.xml");
    defer allocator.free(result2);
    try testing.expect(std.mem.eql(u8, result2, "xl/worksheets/../sharedStrings.xml"));
}
