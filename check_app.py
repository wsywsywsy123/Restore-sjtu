#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import sys

def check_port(port=8501):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

if __name__ == '__main__':
    port = 8501
    if check_port(port):
        print(f"Port {port} is open")
        print(f"App should be running at: http://localhost:{port}")
        print(f"Try opening in browser: http://127.0.0.1:{port}")
    else:
        print(f"Port {port} is not open")
        print("App is not running. Please start it first.")

