#!/usr/bin/env python3
"""
Test script to verify FastAPI app loading and endpoint registration
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing FastAPI app loading...")
    
    # Test imports
    print("1. Testing imports...")
    from main import app
    print("   ✅ Main app imported successfully")
    
    # Check if rules endpoint exists
    print("2. Checking endpoints...")
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append(f"{route.methods} {route.path}")
    
    print(f"   Found {len(routes)} routes:")
    for route in routes:
        print(f"   - {route}")
    
    # Check specifically for rules endpoint
    rules_endpoint = None
    for route in app.routes:
        if hasattr(route, 'path') and route.path == "/rules/":
            rules_endpoint = route
            break
    
    if rules_endpoint:
        print("   ✅ Rules endpoint found!")
        print(f"   - Methods: {rules_endpoint.methods}")
        print(f"   - Tags: {getattr(rules_endpoint, 'tags', 'No tags')}")
    else:
        print("   ❌ Rules endpoint NOT found!")
    
    # Test app startup
    print("3. Testing app startup...")
    from fastapi.testclient import TestClient
    client = TestClient(app)
    print("   ✅ TestClient created successfully")
    
    # Test root endpoint
    print("4. Testing root endpoint...")
    response = client.get("/")
    if response.status_code == 200:
        print("   ✅ Root endpoint working")
        data = response.json()
        if "rules" in data.get("endpoints", {}):
            print("   ✅ Rules endpoint listed in root response")
        else:
            print("   ❌ Rules endpoint NOT listed in root response")
            print(f"   Available endpoints: {data.get('endpoints', {})}")
    else:
        print(f"   ❌ Root endpoint failed: {response.status_code}")
    
    print("\n✅ App loading test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
