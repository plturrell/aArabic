#!/bin/bash
# Script to fix ArrayList API for Zig 0.15.2
# Changes: ArrayList(T).init(allocator) -> ArrayList(T){}
#          .append(item) -> .append(allocator, item)
#          .deinit() -> .deinit(allocator)

cd "$(dirname "$0")"

echo "Fixing ArrayList API in Day 28 components..."

# Fix text_splitter.zig
echo "Fixing text_splitter.zig..."
sed -i.bak 's/ArrayList(\([^)]*\))\.init(self\.allocator)/ArrayList(\1){}/g' components/langflow/text_splitter.zig
sed -i.bak 's/ArrayList(\([^)]*\))\.init(allocator)/ArrayList(\1){}/g' components/langflow/text_splitter.zig
sed -i.bak 's/\.append(\([^,]*\),/\.append(self.allocator, \1,/g' components/langflow/text_splitter.zig
sed -i.bak 's/\.append(\([^)]*\))/\.append(self.allocator, \1)/g' components/langflow/text_splitter.zig
sed -i.bak 's/chunks\.deinit()/chunks.deinit(self.allocator)/g' components/langflow/text_splitter.zig
sed -i.bak 's/result\.deinit()/result.deinit(self.allocator)/g' components/langflow/text_splitter.zig

# Fix file_utils.zig  
echo "Fixing file_utils.zig..."
sed -i.bak 's/ArrayList(\([^)]*\))\.init(self\.allocator)/ArrayList(\1){}/g' components/langflow/file_utils.zig
sed -i.bak 's/ArrayList(\([^)]*\))\.init(allocator)/ArrayList(\1){}/g' components/langflow/file_utils.zig
sed -i.bak 's/\.append(\([^,]*\),/\.append(self.allocator, \1,/g' components/langflow/file_utils.zig
sed -i.bak 's/\.append(\([^)]*\))/\.append(self.allocator, \1)/g' components/langflow/file_utils.zig
sed -i.bak 's/lines\.deinit()/lines.deinit(self.allocator)/g' components/langflow/file_utils.zig
sed -i.bak 's/rows\.deinit()/rows.deinit(self.allocator)/g' components/langflow/file_utils.zig
sed -i.bak 's/fields\.deinit()/fields.deinit(self.allocator)/g' components/langflow/file_utils.zig

# Fix control_flow.zig (if needed)
echo "Fixing control_flow.zig..."
sed -i.bak 's/ArrayList(\([^)]*\))\.init(self\.allocator)/ArrayList(\1){}/g' components/langflow/control_flow.zig
sed -i.bak 's/ArrayList(\([^)]*\))\.init(allocator)/ArrayList(\1){}/g' components/langflow/control_flow.zig

# Remove backup files
rm -f components/langflow/*.bak

echo "Done! Now testing..."
zig test components/langflow/text_splitter.zig
zig test components/langflow/file_utils.zig
zig test components/langflow/control_flow.zig
