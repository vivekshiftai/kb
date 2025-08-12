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
from models.schemas import IoTDeviceRule, MaintenanceData, SafetyPrecaution

logger = structlog.get_logger(__name__)

class RulesGenerator:
    """Service for generating IoT device rules and maintenance data from PDFs"""
    
    def __init__(self):
        self.settings = get_settings()
        self.openai_client = OpenAIClient()
        self.pdf_processor = PDFProcessor()
        
    async def generate_rules_from_pdf(self, pdf_filename: str, chunk_size: int = 30, 
                                    rule_types: List[str] = None, pdf_file_path: str = None) -> Dict[str, Any]:
        """Generate IoT rules, maintenance data, and safety precautions from PDF content using vector database chunks"""
        
        if rule_types is None:
            rule_types = ["monitoring", "maintenance", "alert"]
            
        try:
            logger.info("🤖 Starting IoT rules generation from PDF using vector database", 
                       filename=pdf_filename, 
                       chunk_size=chunk_size, 
                       rule_types=rule_types)
            start_time = datetime.now()
            
            # Import vector store here to avoid circular imports
            from services.vector_store import VectorStore
            vector_store = VectorStore()
            
            # Check if PDF exists in vector database
            logger.info("🔍 Checking if PDF exists in vector database...")
            processed_pdfs = await vector_store.list_processed_pdfs()
            if pdf_filename not in processed_pdfs:
                raise Exception(f"PDF '{pdf_filename}' not found in vector database. Please upload and process the PDF first.")
            logger.info("✅ PDF found in vector database")
            
            # Get all chunks for this PDF
            logger.info("📊 Retrieving all chunks from vector database...")
            all_chunks = await vector_store.get_all_chunks_for_pdf(pdf_filename)
            total_chunks = len(all_chunks)
            logger.info(f"✅ Retrieved {total_chunks} chunks from vector database")
            
            if total_chunks == 0:
                raise Exception(f"No chunks found for PDF '{pdf_filename}' in vector database")
            
            all_rules = []
            all_maintenance_data = []
            all_safety_precautions = []
            processed_batches = 0
            
            # Calculate total batches
            total_batches = (total_chunks + chunk_size - 1) // chunk_size
            logger.info(f"📊 Processing {total_chunks} chunks in {total_batches} batches of {chunk_size} chunks each")
            
            # Process chunks in batches
            for start_idx in range(0, total_chunks, chunk_size):
                end_idx = min(start_idx + chunk_size, total_chunks)
                batch_chunks = all_chunks[start_idx:end_idx]
                batch_num = processed_batches + 1
                
                logger.info(f"🔄 Processing batch {batch_num}/{total_batches}: chunks {start_idx + 1}-{end_idx}")
                
                # Combine text from all chunks in this batch
                batch_text = "\n\n".join([chunk["text"] for chunk in batch_chunks])
                
                if batch_text.strip():
                    logger.info(f"✅ Combined text from batch {batch_num}", text_length=len(batch_text))
                    
                    # Generate rules for this batch
                    logger.info(f"🔧 Generating IoT rules for batch {batch_num}...")
                    batch_rules = await self._generate_batch_rules(batch_text, rule_types, pdf_filename)
                    all_rules.extend(batch_rules)
                    logger.info(f"✅ Generated {len(batch_rules)} rules from batch {batch_num}")
                    
                    # Extract maintenance data for this batch
                    logger.info(f"🔧 Extracting maintenance data for batch {batch_num}...")
                    batch_maintenance = await self._extract_batch_maintenance_data(batch_text, pdf_filename)
                    all_maintenance_data.extend(batch_maintenance)
                    logger.info(f"✅ Extracted {len(batch_maintenance)} maintenance records from batch {batch_num}")
                    
                    # Extract safety precautions for this batch
                    logger.info(f"🔧 Extracting safety precautions for batch {batch_num}...")
                    batch_safety = await self._extract_batch_safety_precautions(batch_text, pdf_filename)
                    all_safety_precautions.extend(batch_safety)
                    logger.info(f"✅ Extracted {len(batch_safety)} safety precautions from batch {batch_num}")
                else:
                    logger.warning(f"⚠️ No text content found in batch {batch_num}")
                
                processed_batches += 1
            
            # Remove duplicates and consolidate
            logger.info("🔍 Removing duplicate rules, maintenance data, and safety precautions...")
            unique_rules = self._deduplicate_rules(all_rules)
            unique_maintenance = self._deduplicate_maintenance(all_maintenance_data)
            unique_safety = self._deduplicate_safety_precautions(all_safety_precautions)
            logger.info(f"✅ Deduplication completed", 
                       original_rules=len(all_rules), 
                       unique_rules=len(unique_rules),
                       original_maintenance=len(all_maintenance_data),
                       unique_maintenance=len(unique_maintenance),
                       original_safety=len(all_safety_precautions),
                       unique_safety=len(unique_safety))
            
            # Generate summary
            logger.info("📝 Generating summary...")
            summary = await self._generate_summary(unique_rules, unique_maintenance, unique_safety, total_chunks)
            logger.info("✅ Summary generated")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"🎉 Rules generation completed successfully", 
                       filename=pdf_filename,
                       total_chunks=total_chunks,
                       processed_batches=processed_batches,
                       total_rules=len(unique_rules),
                       total_maintenance=len(unique_maintenance),
                       total_safety=len(unique_safety),
                       processing_time=f"{processing_time:.2f}s")
            
            return {
                "pdf_filename": pdf_filename,
                "total_pages": total_chunks,  # Using chunks as pages for compatibility
                "processed_chunks": processed_batches,
                "iot_rules": unique_rules,
                "maintenance_data": unique_maintenance,
                "safety_precautions": unique_safety,
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
    
    async def _generate_batch_rules(self, text: str, rule_types: List[str], pdf_filename: str) -> List[IoTDeviceRule]:
        """Generate IoT rules from text batch using OpenAI with strong telemetry-focused prompt"""
        try:
            logger.debug(f"🔧 Creating rules prompt for {len(rule_types)} rule types")
            prompt = self._create_telemetry_rules_prompt(text, rule_types, pdf_filename)
            
            logger.debug("🤖 Calling OpenAI for rules generation...")
            response = await self.openai_client.generate_response(prompt)
            logger.debug("✅ OpenAI response received", response_length=len(response))
            
            # Parse the response to extract rules
            logger.debug("🔍 Parsing OpenAI response for rules...")
            rules = self._parse_rules_response(response)
            logger.debug(f"✅ Parsed {len(rules)} rules from response")
            
            return rules
            
        except Exception as e:
            logger.error(f"❌ Error generating rules from text batch: {e}")
            return []
    
    async def _extract_batch_maintenance_data(self, text: str, pdf_filename: str) -> List[MaintenanceData]:
        """Extract maintenance data from text batch using OpenAI"""
        try:
            logger.debug("🔧 Creating maintenance data prompt")
            prompt = self._create_telemetry_maintenance_prompt(text, pdf_filename)
            
            logger.debug("🤖 Calling OpenAI for maintenance data extraction...")
            response = await self.openai_client.generate_response(prompt)
            logger.debug("✅ OpenAI response received", response_length=len(response))
            
            # Parse the response to extract maintenance data
            logger.debug("🔍 Parsing OpenAI response for maintenance data...")
            maintenance_data = self._parse_maintenance_response(response)
            logger.debug(f"✅ Parsed {len(maintenance_data)} maintenance records from response")
            
            return maintenance_data
            
        except Exception as e:
            logger.error(f"❌ Error extracting maintenance data from text batch: {e}")
            return []

    async def _extract_batch_safety_precautions(self, text: str, pdf_filename: str) -> List[SafetyPrecaution]:
        """Extract safety precautions from text batch using OpenAI"""
        try:
            logger.debug("🔧 Creating safety precautions prompt")
            prompt = self._create_safety_precautions_prompt(text, pdf_filename)
            
            logger.debug("🤖 Calling OpenAI for safety precautions extraction...")
            response = await self.openai_client.generate_response(prompt)
            logger.debug("✅ OpenAI response received", response_length=len(response))
            
            # Parse the response to extract safety precautions
            logger.debug("🔍 Parsing OpenAI response for safety precautions...")
            safety_precautions = self._parse_safety_precautions_response(response)
            logger.debug(f"✅ Parsed {len(safety_precautions)} safety precautions from response")
            
            return safety_precautions
            
        except Exception as e:
            logger.error(f"❌ Error extracting safety precautions from text batch: {e}")
            return []
    
    def _create_telemetry_rules_prompt(self, text: str, rule_types: List[str], pdf_filename: str) -> str:
        """Create strong telemetry-focused prompt for generating IoT rules"""
        rule_types_str = ", ".join(rule_types)
        
        return f"""
You are an expert IoT telemetry analyst and device automation specialist. Your task is to analyze the following technical document content from "{pdf_filename}" and generate SPECIFIC, ACTIONABLE IoT device rules based on telemetry data patterns, sensor readings, and device behavior.

Document Content:
{text[:8000]}  # Increased text length for better context

CRITICAL REQUIREMENTS - READ CAREFULLY:
1. Focus EXCLUSIVELY on telemetry-based rules - temperature, pressure, voltage, current, vibration, humidity, flow rate, speed, etc.
2. Use EXACT sensor names, device identifiers, thresholds, and numerical values mentioned in the document
3. Create rules that trigger based on specific telemetry conditions with precise thresholds
4. Include device-specific identifiers, sensor names, and component names from the document
5. Base ALL rules on actual telemetry patterns, sensor data, or device behavior described in the content
6. Do NOT create generic or hypothetical rules - only rules based on actual telemetry mentioned
7. Use the exact numerical values, units, and thresholds mentioned in the document

Generate IoT device rules in the following JSON format:
{{
    "rules": [
        {{
            "device_name": "exact device/sensor/component name from document",
            "rule_type": "{rule_types_str}",
            "condition": "specific telemetry condition with exact values (e.g., 'temperature > 85°C', 'pressure < 2.5 bar', 'vibration > 0.5 g', 'voltage < 220V')",
            "action": "specific action based on telemetry (e.g., 'shutdown device', 'send alert', 'activate cooling', 'switch to backup power')",
            "priority": "low|medium|high|critical",
            "frequency": "hourly|daily|weekly|monthly",
            "description": "detailed description of the telemetry-based rule, its purpose, and the specific sensor/device it monitors"
        }}
    ]
}}

TELEMETRY FOCUS AREAS - LOOK FOR:
- Temperature sensors and thermal management systems
- Pressure sensors and pressure monitoring systems
- Vibration sensors and mechanical health monitoring
- Electrical sensors (voltage, current, power, frequency)
- Environmental sensors (humidity, air quality, CO2, particulate matter)
- Flow sensors (liquid, gas, air flow rates)
- Speed sensors (RPM, velocity, acceleration)
- Performance metrics and KPIs with numerical thresholds
- Safety thresholds and operational limits
- Predictive maintenance indicators
- Alarm conditions and emergency shutdowns
- Calibration requirements and sensor drift
- Power consumption and energy efficiency metrics

EXAMPLES OF GOOD TELEMETRY RULES:
- "Temperature sensor T1 > 90°C triggers cooling system activation"
- "Pressure sensor P2 < 1.5 bar triggers low pressure alert"
- "Vibration sensor V3 > 0.8 g triggers maintenance check"
- "Voltage sensor V1 < 200V triggers backup power switch"

IMPORTANT: 
- Only generate rules that are directly based on telemetry data, sensor readings, or device behavior mentioned in the document
- Do NOT create generic rules or hypothetical scenarios
- Use exact values, thresholds, sensor names, and device identifiers from the content
- If no telemetry data is found, return an empty rules array

Return only valid JSON without any additional text or explanations.
"""
    
    def _create_telemetry_maintenance_prompt(self, text: str, pdf_filename: str) -> str:
        """Create telemetry-focused prompt for extracting maintenance data"""
        return f"""
You are an expert IoT maintenance analyst and predictive maintenance specialist. Analyze the following technical document content from "{pdf_filename}" and extract maintenance information based on telemetry data, sensor readings, and device performance patterns.

Document Content:
{text[:8000]}  # Increased text length for better context

CRITICAL REQUIREMENTS - READ CAREFULLY:
1. Focus EXCLUSIVELY on maintenance schedules based on telemetry indicators and sensor readings
2. Extract maintenance requirements tied to specific sensor readings and thresholds
3. Identify predictive maintenance triggers based on device behavior and performance patterns
4. Use exact component names, sensor names, and telemetry thresholds from the document
5. Base ALL maintenance schedules on actual telemetry patterns, sensor data, or device behavior
6. Do NOT create generic maintenance schedules - only those based on actual telemetry mentioned
7. Use exact numerical values, units, and thresholds mentioned in the document

Extract maintenance data in the following JSON format:
{{
    "maintenance": [
        {{
            "component_name": "exact component/sensor/device name from document",
            "maintenance_type": "preventive|corrective|predictive",
            "frequency": "maintenance frequency based on telemetry (e.g., 'when temperature exceeds 80°C', 'every 1000 operating hours', 'when vibration > 0.5 g')",
            "last_maintenance": "last maintenance reference from document (if mentioned)",
            "next_maintenance": "next maintenance trigger from document (if mentioned)",
            "description": "detailed description of maintenance requirements based on telemetry data and sensor readings"
        }}
    ]
}}

TELEMETRY-BASED MAINTENANCE FOCUS - LOOK FOR:
- Maintenance triggered by specific sensor readings (temperature, pressure, vibration, etc.)
- Predictive maintenance based on performance trends and sensor drift
- Component replacement based on usage metrics and operating hours
- Calibration requirements based on sensor accuracy and drift patterns
- Safety maintenance based on threshold violations and alarm conditions
- Performance-based maintenance schedules tied to efficiency metrics
- Preventive maintenance based on wear indicators and sensor data
- Emergency maintenance triggers based on critical sensor readings

EXAMPLES OF GOOD TELEMETRY-BASED MAINTENANCE:
- "Temperature sensor calibration required when drift exceeds ±2°C"
- "Bearing replacement when vibration sensor exceeds 0.8 g for 24 hours"
- "Filter replacement when pressure differential exceeds 5 bar"
- "Motor maintenance when current consumption increases by 15%"

IMPORTANT: 
- Only extract maintenance data that is directly related to telemetry, sensor readings, or device performance patterns mentioned in the document
- Do NOT create generic maintenance schedules or hypothetical scenarios
- Use exact values, thresholds, sensor names, and device identifiers from the content
- If no telemetry-based maintenance data is found, return an empty maintenance array

Return only valid JSON without any additional text or explanations.
"""

    def _create_safety_precautions_prompt(self, text: str, pdf_filename: str) -> str:
        """Create safety precautions prompt for extracting safety information"""
        return f"""
You are an expert safety analyst and occupational health specialist. Analyze the following technical document content from "{pdf_filename}" and extract safety precautions, warnings, and safety-related information.

Document Content:
{text[:8000]}  # Increased text length for better context

CRITICAL REQUIREMENTS - READ CAREFULLY:
1. Focus EXCLUSIVELY on safety precautions, warnings, and safety-related information
2. Extract safety measures, protective equipment requirements, and hazard warnings
3. Identify emergency procedures, evacuation plans, and safety protocols
4. Use exact safety terms, equipment names, and procedures from the document
5. Base ALL safety precautions on actual safety information mentioned in the content
6. Do NOT create generic safety advice - only extract specific safety information
7. Use exact safety thresholds, procedures, and equipment mentioned in the document

Extract safety precautions in the following JSON format:
{{
    "safety_precautions": [
        {{
            "category": "safety category (e.g., 'electrical safety', 'mechanical safety', 'chemical safety', 'fire safety', 'personal protection')",
            "precaution": "specific safety precaution or warning from document",
            "equipment": "required safety equipment or PPE (if mentioned)",
            "procedure": "safety procedure or protocol (if mentioned)",
            "warning_level": "low|medium|high|critical",
            "description": "detailed description of the safety precaution and its importance"
        }}
    ]
}}

SAFETY FOCUS AREAS - LOOK FOR:
- Electrical safety precautions and warnings
- Mechanical safety and equipment protection
- Chemical safety and hazardous material handling
- Fire safety and emergency procedures
- Personal protective equipment (PPE) requirements
- Environmental safety and pollution prevention
- Emergency shutdown procedures
- Lockout/tagout procedures
- Ventilation and air quality requirements
- Noise and vibration protection
- Radiation safety (if applicable)
- Biological safety (if applicable)

EXAMPLES OF GOOD SAFETY PRECAUTIONS:
- "Electrical safety: Disconnect power before maintenance, use insulated tools"
- "Fire safety: Keep flammable materials away from heat sources, install smoke detectors"
- "Personal protection: Wear safety goggles and gloves when handling chemicals"
- "Emergency procedures: Know location of emergency exits and fire extinguishers"

IMPORTANT: 
- Only extract safety precautions that are directly mentioned in the document
- Do NOT create generic safety advice or hypothetical scenarios
- Use exact safety terms, equipment names, and procedures from the content
- If no safety information is found, return an empty safety_precautions array

Return only valid JSON without any additional text or explanations.
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
    
    def _parse_safety_precautions_response(self, response: str) -> List[SafetyPrecaution]:
        """Parse OpenAI response to extract safety precautions"""
        try:
            import json
            
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                
                data = json.loads(json_str)
                safety_data = data.get("safety_precautions", [])
                
                safety_precautions = []
                for safety_item in safety_data:
                    try:
                        safety = SafetyPrecaution(
                            category=safety_item.get("category", "General Safety"),
                            precaution=safety_item.get("precaution", ""),
                            equipment=safety_item.get("equipment"),
                            procedure=safety_item.get("procedure"),
                            warning_level=safety_item.get("warning_level", "medium"),
                            description=safety_item.get("description", "")
                        )
                        safety_precautions.append(safety)
                    except Exception as e:
                        logger.warning(f"Error parsing safety precaution: {e}")
                
                return safety_precautions
            
            return []
            
        except Exception as e:
            logger.error(f"Error parsing safety precautions response: {e}")
            return []
    
    def _deduplicate_safety_precautions(self, safety_precautions: List[SafetyPrecaution]) -> List[SafetyPrecaution]:
        """Remove duplicate safety precautions based on category and precaution"""
        seen = set()
        unique_safety = []
        
        for safety in safety_precautions:
            key = f"{safety.category}_{safety.precaution}"
            if key not in seen:
                seen.add(key)
                unique_safety.append(safety)
        
        return unique_safety
    
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
                              safety_precautions: List[SafetyPrecaution],
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

Safety Precautions: {len(safety_precautions)}
- Critical: {len([s for s in safety_precautions if s.warning_level == 'critical'])}
- High: {len([s for s in safety_precautions if s.warning_level == 'high'])}
- Medium: {len([s for s in safety_precautions if s.warning_level == 'medium'])}
- Low: {len([s for s in safety_precautions if s.warning_level == 'low'])}

Generate a concise summary (2-3 sentences) of the key IoT, maintenance, and safety insights from this document.
"""
            
            response = await self.openai_client.generate_response(summary_prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Successfully analyzed {total_pages} pages and generated {len(rules)} IoT rules, {len(maintenance)} maintenance records, and {len(safety_precautions)} safety precautions."
