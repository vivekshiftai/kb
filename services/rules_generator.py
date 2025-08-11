"""
Rules Generator Service
Generates IoT device rules and maintenance data from PDF content
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import fitz  # PyMuPDF
import structlog

from config.settings import get_settings
from services.openai_client import OpenAIClient
from services.pdf_processor import PDFProcessor
from models.schemas import IoTDeviceRule, MaintenanceData

logger = structlog.get_logger(__name__)

class RulesGenerator:
    """Service for generating IoT device rules and maintenance data from PDFs"""
    
    def __init__(self):
        self.settings = get_settings()
        self.openai_client = OpenAIClient()
        self.pdf_processor = PDFProcessor()
        
    async def generate_rules_from_pdf(self, pdf_filename: str, chunk_size: int = 10, 
                                    rule_types: List[str] = None, pdf_file_path: str = None) -> Dict[str, Any]:
        """Generate IoT rules and maintenance data from PDF content in chunks"""
        
        if rule_types is None:
            rule_types = ["monitoring", "maintenance", "alert"]
            
        try:
            logger.info("🤖 Starting IoT rules generation from PDF", 
                       filename=pdf_filename, 
                       chunk_size=chunk_size, 
                       rule_types=rule_types)
            start_time = datetime.now()
            
            # Get PDF file path - support both uploaded files and direct file paths
            logger.info("🔍 Determining PDF file path...")
            if pdf_file_path:
                pdf_path = pdf_file_path
                logger.info("✅ Using provided file path", path=pdf_file_path)
            else:
                pdf_path = f"{self.settings.UPLOAD_DIR}/{pdf_filename}"
                logger.info("✅ Using upload directory path", path=pdf_path)
            
            # Open PDF and get total pages
            logger.info("📄 Opening PDF document...")
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            logger.info(f"✅ PDF opened successfully", total_pages=total_pages, filename=pdf_filename)
            
            all_rules = []
            all_maintenance_data = []
            processed_chunks = 0
            
            # Calculate total chunks
            total_chunks = (total_pages + chunk_size - 1) // chunk_size
            logger.info(f"📊 Processing {total_pages} pages in {total_chunks} chunks of {chunk_size} pages each")
            
            # Process PDF in chunks
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                chunk_pages = list(range(start_page, end_page))
                chunk_num = processed_chunks + 1
                
                logger.info(f"🔄 Processing chunk {chunk_num}/{total_chunks}: pages {start_page + 1}-{end_page}")
                
                # Extract text from this chunk
                logger.info(f"📝 Extracting text from chunk {chunk_num}...")
                chunk_text = await self._extract_chunk_text(doc, chunk_pages)
                
                if chunk_text.strip():
                    logger.info(f"✅ Text extracted from chunk {chunk_num}", text_length=len(chunk_text))
                    
                    # Generate rules for this chunk
                    logger.info(f"🔧 Generating IoT rules for chunk {chunk_num}...")
                    chunk_rules = await self._generate_chunk_rules(chunk_text, rule_types)
                    all_rules.extend(chunk_rules)
                    logger.info(f"✅ Generated {len(chunk_rules)} rules from chunk {chunk_num}")
                    
                    # Extract maintenance data for this chunk
                    logger.info(f"🔧 Extracting maintenance data for chunk {chunk_num}...")
                    chunk_maintenance = await self._extract_maintenance_data(chunk_text)
                    all_maintenance_data.extend(chunk_maintenance)
                    logger.info(f"✅ Extracted {len(chunk_maintenance)} maintenance records from chunk {chunk_num}")
                else:
                    logger.warning(f"⚠️ No text content found in chunk {chunk_num}")
                
                processed_chunks += 1
                
            doc.close()
            logger.info("✅ PDF document closed")
            
            # Remove duplicates and consolidate
            logger.info("🔍 Removing duplicate rules and maintenance data...")
            unique_rules = self._deduplicate_rules(all_rules)
            unique_maintenance = self._deduplicate_maintenance(all_maintenance_data)
            logger.info(f"✅ Deduplication completed", 
                       original_rules=len(all_rules), 
                       unique_rules=len(unique_rules),
                       original_maintenance=len(all_maintenance_data),
                       unique_maintenance=len(unique_maintenance))
            
            # Generate summary
            logger.info("📝 Generating summary...")
            summary = await self._generate_summary(unique_rules, unique_maintenance, total_pages)
            logger.info("✅ Summary generated")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"🎉 Rules generation completed successfully", 
                       filename=pdf_filename,
                       total_pages=total_pages,
                       processed_chunks=processed_chunks,
                       total_rules=len(unique_rules),
                       total_maintenance=len(unique_maintenance),
                       processing_time=f"{processing_time:.2f}s")
            
            return {
                "pdf_filename": pdf_filename,
                "total_pages": total_pages,
                "processed_chunks": processed_chunks,
                "iot_rules": unique_rules,
                "maintenance_data": unique_maintenance,
                "processing_time": processing_time,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating rules from PDF {pdf_filename}: {e}")
            import traceback
            logger.error("❌ Full error traceback:", traceback=traceback.format_exc())
            raise
    
    async def _extract_chunk_text(self, doc: fitz.Document, page_numbers: List[int]) -> str:
        """Extract text from specific pages"""
        try:
            logger.debug(f"📝 Extracting text from pages: {[p+1 for p in page_numbers]}")
            text_chunks = []
            
            for page_num in page_numbers:
                page = doc[page_num]
                text = page.get_text()
                text_chunks.append(f"Page {page_num + 1}:\n{text}")
                logger.debug(f"📄 Page {page_num + 1}: {len(text)} characters")
            
            combined_text = "\n\n".join(text_chunks)
            logger.debug(f"📊 Total text extracted: {len(combined_text)} characters")
            return combined_text
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from pages {page_numbers}: {e}")
            return ""
    
    async def _generate_chunk_rules(self, text: str, rule_types: List[str]) -> List[IoTDeviceRule]:
        """Generate IoT rules from text chunk using OpenAI"""
        try:
            logger.debug(f"🔧 Creating rules prompt for {len(rule_types)} rule types")
            prompt = self._create_rules_prompt(text, rule_types)
            
            logger.debug("🤖 Calling OpenAI for rules generation...")
            response = await self.openai_client.generate_response(prompt)
            logger.debug("✅ OpenAI response received", response_length=len(response))
            
            # Parse the response to extract rules
            logger.debug("🔍 Parsing OpenAI response for rules...")
            rules = self._parse_rules_response(response)
            logger.debug(f"✅ Parsed {len(rules)} rules from response")
            
            return rules
            
        except Exception as e:
            logger.error(f"❌ Error generating rules from text chunk: {e}")
            return []
    
    async def _extract_maintenance_data(self, text: str) -> List[MaintenanceData]:
        """Extract maintenance data from text chunk using OpenAI"""
        try:
            logger.debug("🔧 Creating maintenance data prompt")
            prompt = self._create_maintenance_prompt(text)
            
            logger.debug("🤖 Calling OpenAI for maintenance data extraction...")
            response = await self.openai_client.generate_response(prompt)
            logger.debug("✅ OpenAI response received", response_length=len(response))
            
            # Parse the response to extract maintenance data
            logger.debug("🔍 Parsing OpenAI response for maintenance data...")
            maintenance_data = self._parse_maintenance_response(response)
            logger.debug(f"✅ Parsed {len(maintenance_data)} maintenance records from response")
            
            return maintenance_data
            
        except Exception as e:
            logger.error(f"❌ Error extracting maintenance data from text chunk: {e}")
            return []
    
    def _create_rules_prompt(self, text: str, rule_types: List[str]) -> str:
        """Create prompt for generating IoT rules"""
        rule_types_str = ", ".join(rule_types)
        
        return f"""
Analyze the following technical document content and generate IoT device rules. Focus on {rule_types_str} rules.

Document Content:
{text[:4000]}  # Limit text length for API

Generate IoT device rules in the following JSON format:
{{
    "rules": [
        {{
            "device_name": "device identifier",
            "rule_type": "monitoring|maintenance|alert|control",
            "condition": "specific condition or threshold",
            "action": "action to take when condition is met",
            "priority": "low|medium|high|critical",
            "frequency": "hourly|daily|weekly|monthly",
            "description": "detailed description of the rule"
        }}
    ]
}}

Focus on:
- Device monitoring and control rules
- Alert conditions and thresholds
- Maintenance schedules and requirements
- Safety and operational rules
- Performance optimization rules

Return only valid JSON without any additional text.
"""
    
    def _create_maintenance_prompt(self, text: str) -> str:
        """Create prompt for extracting maintenance data"""
        return f"""
Analyze the following technical document content and extract maintenance information.

Document Content:
{text[:4000]}  # Limit text length for API

Extract maintenance data in the following JSON format:
{{
    "maintenance": [
        {{
            "component_name": "name of the component or system",
            "maintenance_type": "preventive|corrective|predictive",
            "frequency": "how often maintenance should be performed",
            "last_maintenance": "when last maintenance was done (if mentioned)",
            "next_maintenance": "when next maintenance is due (if mentioned)",
            "description": "detailed description of maintenance requirements"
        }}
    ]
}}

Focus on:
- Preventive maintenance schedules
- Component replacement intervals
- Inspection requirements
- Calibration needs
- Cleaning and lubrication schedules
- Safety checks and certifications

Return only valid JSON without any additional text.
"""
    
    def _parse_rules_response(self, response: str) -> List[IoTDeviceRule]:
        """Parse OpenAI response to extract IoT rules"""
        try:
            import json
            
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                
                data = json.loads(json_str)
                rules_data = data.get("rules", [])
                
                rules = []
                for rule_data in rules_data:
                    try:
                        rule = IoTDeviceRule(
                            device_name=rule_data.get("device_name", "Unknown Device"),
                            rule_type=rule_data.get("rule_type", "monitoring"),
                            condition=rule_data.get("condition", ""),
                            action=rule_data.get("action", ""),
                            priority=rule_data.get("priority", "medium"),
                            frequency=rule_data.get("frequency"),
                            description=rule_data.get("description", "")
                        )
                        rules.append(rule)
                    except Exception as e:
                        logger.warning(f"Error parsing rule: {e}")
                
                return rules
            
            return []
            
        except Exception as e:
            logger.error(f"Error parsing rules response: {e}")
            return []
    
    def _parse_maintenance_response(self, response: str) -> List[MaintenanceData]:
        """Parse OpenAI response to extract maintenance data"""
        try:
            import json
            
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                
                data = json.loads(json_str)
                maintenance_data = data.get("maintenance", [])
                
                maintenance = []
                for maint_data in maintenance_data:
                    try:
                        maint = MaintenanceData(
                            component_name=maint_data.get("component_name", "Unknown Component"),
                            maintenance_type=maint_data.get("maintenance_type", "preventive"),
                            frequency=maint_data.get("frequency", ""),
                            last_maintenance=maint_data.get("last_maintenance"),
                            next_maintenance=maint_data.get("next_maintenance"),
                            description=maint_data.get("description", "")
                        )
                        maintenance.append(maint)
                    except Exception as e:
                        logger.warning(f"Error parsing maintenance data: {e}")
                
                return maintenance
            
            return []
            
        except Exception as e:
            logger.error(f"Error parsing maintenance response: {e}")
            return []
    
    def _deduplicate_rules(self, rules: List[IoTDeviceRule]) -> List[IoTDeviceRule]:
        """Remove duplicate rules based on device name and condition"""
        seen = set()
        unique_rules = []
        
        for rule in rules:
            key = f"{rule.device_name}_{rule.condition}_{rule.action}"
            if key not in seen:
                seen.add(key)
                unique_rules.append(rule)
        
        return unique_rules
    
    def _deduplicate_maintenance(self, maintenance: List[MaintenanceData]) -> List[MaintenanceData]:
        """Remove duplicate maintenance data based on component name and type"""
        seen = set()
        unique_maintenance = []
        
        for maint in maintenance:
            key = f"{maint.component_name}_{maint.maintenance_type}_{maint.frequency}"
            if key not in seen:
                seen.add(key)
                unique_maintenance.append(maint)
        
        return unique_maintenance
    
    async def _generate_summary(self, rules: List[IoTDeviceRule], 
                              maintenance: List[MaintenanceData], 
                              total_pages: int) -> str:
        """Generate a summary of the analysis"""
        try:
            summary_prompt = f"""
Based on the analysis of a {total_pages}-page technical document, here are the key findings:

IoT Rules Generated: {len(rules)}
- Monitoring rules: {len([r for r in rules if r.rule_type == 'monitoring'])}
- Maintenance rules: {len([r for r in rules if r.rule_type == 'maintenance'])}
- Alert rules: {len([r for r in rules if r.rule_type == 'alert'])}
- Control rules: {len([r for r in rules if r.rule_type == 'control'])}

Maintenance Records: {len(maintenance)}
- Preventive: {len([m for m in maintenance if m.maintenance_type == 'preventive'])}
- Corrective: {len([m for m in maintenance if m.maintenance_type == 'corrective'])}
- Predictive: {len([m for m in maintenance if m.maintenance_type == 'predictive'])}

Generate a concise summary (2-3 sentences) of the key IoT and maintenance insights from this document.
"""
            
            response = await self.openai_client.generate_response(summary_prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Successfully analyzed {total_pages} pages and generated {len(rules)} IoT rules and {len(maintenance)} maintenance records."
