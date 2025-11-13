#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose app issues"""
import socket
import subprocess
import sys
import os

def test_connection(port=8501):
    print("=" * 60)
    print("App Diagnosis Tool")
    print("=" * 60)
    
    # Test 1: Port check
    print("\n[1] Checking port status...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    
    if result == 0:
        print(f"    OK: Port {port} is open and listening")
    else:
        print(f"    FAIL: Port {port} is not accessible")
        print("    Solution: Run 'python start_app.py' to start the app")
        return
    
    # Test 2: Try HTTP connection
    print("\n[2] Testing HTTP connection...")
    try:
        import urllib.request
        response = urllib.request.urlopen(f'http://localhost:{port}', timeout=5)
        print(f"    OK: HTTP connection successful (Status: {response.getcode()})")
    except Exception as e:
        print(f"    WARNING: HTTP connection failed: {e}")
        print("    This might be normal if app is still starting")
    
    # Test 3: Browser suggestions
    print("\n[3] Browser Access URLs:")
    print(f"    - http://localhost:{port}")
    print(f"    - http://127.0.0.1:{port}")
    print(f"    - http://0.0.0.0:{port}")
    
    # Test 4: Common issues
    print("\n[4] Common Issues & Solutions:")
    print("    If browser shows 'Unable to connect':")
    print("    1. Wait 10-30 seconds for app to fully start")
    print("    2. Try refreshing the page")
    print("    3. Check Windows Firewall settings")
    print("    4. Try using 127.0.0.1 instead of localhost")
    print("    5. Check if antivirus is blocking the connection")
    
    print("\n" + "=" * 60)
    print("Diagnosis complete!")
    print("=" * 60)

if __name__ == '__main__':
    port = 8501
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except:
            pass
    test_connection(port)

