"""PhotoCraft の操作用クライアント（photocraft --control 7878 で起動しておく）。

  python tools/pc.py <method> '<params json>'
  例: python tools/pc.py app.open '{"path":"4_4.png"}'
"""
import json, socket, sys, base64, pathlib

TOKEN = pathlib.Path(r"D:\tools\photocraft\control.token").read_text().strip()

class PC:
    def __init__(self, port=7878):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=300)
        self.f = self.s.makefile("rwb")
        self.n = 0
        self.call("auth", {"token": TOKEN})

    def call(self, method, params=None):
        self.n += 1
        rid = self.n
        self.f.write((json.dumps({"id": rid, "method": method, "params": params or {}}) + "\n").encode())
        self.f.flush()
        while True:
            r = json.loads(self.f.readline())
            if r.get("id") == rid:
                if not r.get("ok"):
                    raise RuntimeError(r.get("error"))
                return r.get("result")

    def run(self, command, params=None):
        return self.call("engine.execute", {"command": command, "params": params or {}})

    def screenshot(self, path):
        r = self.call("ui.screenshot", {})
        data = r.get("data") or r.get("png") or r.get("base64")
        pathlib.Path(path).write_bytes(base64.b64decode(data))

if __name__ == "__main__":
    pc = PC()
    out = pc.call(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
    print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])
