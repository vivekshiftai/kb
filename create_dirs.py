#!/usr/bin/env python3
import os

# Create required directories
directories = ['uploads', 'outputs', 'chroma_db']

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✅ Created directory: {directory}")

print("\n🎉 All directories created successfully!")
print("You can now run the application with: uvicorn main:app --reload")
