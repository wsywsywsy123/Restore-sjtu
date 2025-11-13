#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import sys
import os
import socket
import time

def kill_port(port):
    if os.name != 'nt':
        return False
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) > 4:
                    pid = parts[-1]
                    subprocess.run(['taskkill', '/F', '/PID', pid], 
                                 capture_output=True)
                    time.sleep(2)
                    return True
    except:
        pass
    return False

def check_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

if __name__ == '__main__':
    port = 8501
    
    if check_port(port):
        print(f"Port {port} is in use. Killing process...")
        kill_port(port)
        time.sleep(2)
    
    print(f"Starting app on port {port}...")
    print(f"Open browser: http://localhost:{port}")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py',
                       '--server.port', str(port)])
    except KeyboardInterrupt:
        print("\nStopped")
