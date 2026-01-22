"""
Simple JSON Builder for Mojo.
"""

from collections import List

struct JsonBuilder:
    var parts: List[String]

    fn __init__(out self):
        self.parts = List[String]()

    fn append(mut self, s: String):
        self.parts.append(s)

    fn begin_object(mut self):
        self.parts.append("{")

    fn end_object(mut self):
        self.parts.append("}")

    fn begin_array(mut self):
        self.parts.append("[")

    fn end_array(mut self):
        self.parts.append("]")

    fn key(mut self, k: String):
        self.parts.append('"' + k + '":')

    fn comma(mut self):
        self.parts.append(",")

    fn value_bool(mut self, b: Bool):
        if b:
            self.parts.append("true")
        else:
            self.parts.append("false")

    fn value_int(mut self, i: Int):
        self.parts.append(str(i))

    fn value_string(mut self, s: String):
        self.parts.append('"' + self.escape(s) + '"')

    fn value_null(mut self):
        self.parts.append("null")

    fn escape(self, s: String) -> String:
        var res = String("")
        # Basic escaping (expand as needed)
        for i in range(len(s)):
            var c = s[i]
            if c == '"': res += '\"'
            elif c == '\\': res += '\\\\'
            elif c == '\n': res += '\\n'
            elif c == '\r': res += '\\r'
            elif c == '\t': res += '\\t'
            else: res += c
        return res

    fn to_string(self) -> String:
        var res = String("")
        for i in range(len(self.parts)):
            res += self.parts[i]
        return res
