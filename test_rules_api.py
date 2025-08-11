#!/usr/bin/env python3
"""
Test script for the Rules API functionality
"""

import asyncio
import json
from services.rules_generator import RulesGenerator
from config.settings import get_settings

async def test_rules_generation():
    """Test the rules generation functionality"""
    
    print("🧪 Testing Rules Generation API...")
    print("=" * 50)
    
    # Initialize the rules generator
    rules_generator = RulesGenerator()
    
    # Test parameters
    test_pdf = "your_test_document.pdf"  # Replace with actual PDF filename
    chunk_size = 10
    rule_types = ["monitoring", "maintenance", "alert"]
    
    try:
        print(f"📄 Processing PDF: {test_pdf}")
        print(f"📊 Chunk size: {chunk_size} pages")
        print(f"🎯 Rule types: {', '.join(rule_types)}")
        print()
        
        # Generate rules
        result = await rules_generator.generate_rules_from_pdf(
            pdf_filename=test_pdf,
            chunk_size=chunk_size,
            rule_types=rule_types
        )
        
        # Display results
        print("✅ Rules generation completed successfully!")
        print(f"📈 Total pages processed: {result['total_pages']}")
        print(f"🔄 Chunks processed: {result['processed_chunks']}")
        print(f"⚙️  IoT rules generated: {len(result['iot_rules'])}")
        print(f"🔧 Maintenance records: {len(result['maintenance_data'])}")
        print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
        print()
        
        # Display summary
        print("📋 Summary:")
        print(result['summary'])
        print()
        
        # Display sample rules
        if result['iot_rules']:
            print("🔍 Sample IoT Rules:")
            for i, rule in enumerate(result['iot_rules'][:3]):  # Show first 3 rules
                print(f"  {i+1}. {rule.device_name} - {rule.rule_type}")
                print(f"     Condition: {rule.condition}")
                print(f"     Action: {rule.action}")
                print(f"     Priority: {rule.priority}")
                print()
        
        # Display sample maintenance data
        if result['maintenance_data']:
            print("🔧 Sample Maintenance Data:")
            for i, maint in enumerate(result['maintenance_data'][:3]):  # Show first 3 records
                print(f"  {i+1}. {maint.component_name} - {maint.maintenance_type}")
                print(f"     Frequency: {maint.frequency}")
                print(f"     Description: {maint.description}")
                print()
        
        print("🎉 Test completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: PDF file '{test_pdf}' not found!")
        print("Please upload a PDF file first using the upload endpoint.")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Please check the application logs for more details.")

def test_api_endpoint():
    """Test the API endpoint using requests"""
    import requests
    
    print("🌐 Testing Rules API Endpoint...")
    print("=" * 50)
    
    # API endpoint
    url = "http://localhost:8000/rules/"
    
    # Test request
    payload = {
        "pdf_filename": "your_test_document.pdf",  # Replace with actual PDF filename
        "chunk_size": 10,
        "rule_types": ["monitoring", "maintenance", "alert"]
    }
    
    try:
        print(f"📡 Sending request to: {url}")
        print(f"📄 PDF: {payload['pdf_filename']}")
        print()
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"📈 Total pages: {result['total_pages']}")
            print(f"⚙️  IoT rules: {len(result['iot_rules'])}")
            print(f"🔧 Maintenance records: {len(result['maintenance_data'])}")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            print()
            print("📋 Summary:")
            print(result['summary'])
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server!")
        print("Make sure the application is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error during API test: {e}")

if __name__ == "__main__":
    print("🚀 Rules API Test Suite")
    print("=" * 60)
    print()
    
    # Test 1: Direct service test
    print("Test 1: Direct Service Test")
    print("-" * 30)
    asyncio.run(test_rules_generation())
    print()
    
    # Test 2: API endpoint test
    print("Test 2: API Endpoint Test")
    print("-" * 30)
    test_api_endpoint()
    print()
    
    print("🏁 All tests completed!")
