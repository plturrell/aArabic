const std = @import("std");
const hf = @import("huggingface_loader");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  DYNAMIC MODEL DISCOVERY & TESTING\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    const base_path = "/Users/user/Documents/arabic_folder/vendor/layerModels/huggingFace";
    
    std.debug.print("\n🔍 Scanning for HuggingFace models in: {s}\n", .{base_path});
    
    // Open base directory
    var base_dir = try std.fs.openDirAbsolute(base_path, .{ .iterate = true });
    defer base_dir.close();
    
    var vendor_iter = base_dir.iterate();
    var models_found: usize = 0;
    var models_tested: usize = 0;
    var models_passed: usize = 0;
    
    // Iterate through vendor directories
    while (try vendor_iter.next()) |vendor_entry| {
        if (vendor_entry.kind != .directory) continue;
        if (std.mem.eql(u8, vendor_entry.name, ".DS_Store")) continue;
        
        std.debug.print("\n📁 Vendor: {s}\n", .{vendor_entry.name});
        
        // Open vendor directory
        const vendor_path = try std.fs.path.join(allocator, &[_][]const u8{ base_path, vendor_entry.name });
        defer allocator.free(vendor_path);
        
        var vendor_dir = try std.fs.openDirAbsolute(vendor_path, .{ .iterate = true });
        defer vendor_dir.close();
        
        var model_iter = vendor_dir.iterate();
        
        // Iterate through models in vendor directory
        while (try model_iter.next()) |model_entry| {
            if (model_entry.kind != .directory) continue;
            if (std.mem.eql(u8, model_entry.name, ".DS_Store")) continue;
            
            const model_path = try std.fs.path.join(allocator, &[_][]const u8{ vendor_path, model_entry.name });
            defer allocator.free(model_path);
            
            models_found += 1;
            
            std.debug.print("   🔸 Model: {s}\n", .{model_entry.name});
            std.debug.print("      Path: {s}\n", .{model_path});
            
            // Check if it's a valid HuggingFace model
            if (hf.isHuggingFaceModel(allocator, model_path)) {
                std.debug.print("      ✅ Valid HuggingFace model detected\n", .{});
                models_tested += 1;
                
                // Try to load the model
                std.debug.print("      🔄 Loading model...\n", .{});
                
                var model = hf.HuggingFaceModel.init(allocator, model_path);
                defer model.deinit();
                
                if (model.load()) {
                    std.debug.print("      ✅ Model loaded successfully!\n", .{});
                    std.debug.print("         Architecture: {s}\n", .{@tagName(model.config.architecture)});
                    std.debug.print("         Layers: {d}\n", .{model.config.num_hidden_layers});
                    std.debug.print("         Hidden size: {d}\n", .{model.config.hidden_size});
                    std.debug.print("         Vocab: {d}\n", .{model.config.vocab_size});
                    std.debug.print("         Tensors: {d}\n", .{model.weights.index.weight_map.count()});
                    
                    models_passed += 1;
                } else |err| {
                    std.debug.print("      ⚠️  Load failed: {}\n", .{err});
                }
            } else {
                std.debug.print("      ⚠️  Not a compatible HuggingFace model (missing vocab.json or config.json)\n", .{});
            }
        }
    }
    
    // Summary
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  DISCOVERY SUMMARY\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n📊 Results:\n", .{});
    std.debug.print("   Models found: {d}\n", .{models_found});
    std.debug.print("   Models tested: {d}\n", .{models_tested});
    std.debug.print("   Models passed: {d}\n", .{models_passed});
    
    if (models_passed > 0) {
        std.debug.print("\n✅ Dynamic model discovery working!\n", .{});
    } else if (models_tested > 0) {
        std.debug.print("\n⚠️  Models detected but loading failed (may need weight files)\n", .{});
    } else {
        std.debug.print("\n⚠️  No compatible models found\n", .{});
    }
    
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
}
