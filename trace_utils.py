#!/usr/bin/env python3
"""
Trace management utilities for tau-bench.
Provides command-line tools for managing, exporting, and analyzing traces.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add tau_bench to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tau_bench.agents.trace_exporters import (
    TraceExportConfig,
    create_trace_export_presets,
    get_trace_files,
    merge_trace_files,
    export_traces_from_phoenix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_traces(traces_dir: str = "./traces") -> None:
    """List all available trace files."""
    trace_files = get_trace_files(traces_dir)
    
    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        return
    
    print(f"Found {len(trace_files)} trace files in {traces_dir}:")
    print("-" * 60)
    
    for trace_file in trace_files:
        try:
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
            
            span_count = len(trace_data.get("spans", []))
            timestamp = trace_data.get("timestamp", "unknown")
            file_size = os.path.getsize(trace_file)
            
            print(f"📄 {os.path.basename(trace_file)}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Spans: {span_count}")
            print(f"   Size: {file_size:,} bytes")
            print()
            
        except Exception as e:
            print(f"❌ {os.path.basename(trace_file)} (error reading: {e})")


def analyze_trace(trace_file: str) -> None:
    """Analyze a single trace file and show detailed statistics."""
    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        spans = trace_data.get("spans", [])
        
        print(f"📊 Trace Analysis: {os.path.basename(trace_file)}")
        print("=" * 60)
        print(f"Total spans: {len(spans)}")
        print(f"Timestamp: {trace_data.get('timestamp', 'unknown')}")
        
        if not spans:
            print("No spans found in trace file")
            return
        
        # Analyze span names
        span_names = {}
        total_duration = 0
        llm_calls = 0
        tool_calls = 0
        
        for span in spans:
            name = span.get("name", "unknown")
            span_names[name] = span_names.get(name, 0) + 1
            
            # Calculate duration if available
            duration = span.get("duration_ns")
            if duration:
                total_duration += duration
            
            # Count specific span types
            if "llm" in name.lower():
                llm_calls += 1
            elif "tool" in name.lower():
                tool_calls += 1
        
        print(f"\n📈 Span Statistics:")
        print(f"LLM calls: {llm_calls}")
        print(f"Tool calls: {tool_calls}")
        print(f"Total duration: {total_duration / 1e9:.2f} seconds" if total_duration else "Duration: N/A")
        
        print(f"\n🏷️  Span Types:")
        for name, count in sorted(span_names.items()):
            print(f"  {name}: {count}")
        
        # Analyze attributes
        all_attributes = set()
        for span in spans:
            attributes = span.get("attributes", {})
            all_attributes.update(attributes.keys())
        
        if all_attributes:
            print(f"\n🔍 Available Attributes:")
            for attr in sorted(all_attributes):
                print(f"  {attr}")
        
        # Show trace structure (simplified)
        print(f"\n🌳 Trace Structure (first 10 spans):")
        for i, span in enumerate(spans[:10]):
            name = span.get("name", "unknown")
            parent_id = span.get("parent_span_id")
            indent = "  " if parent_id else ""
            print(f"  {i+1:2d}. {indent}{name}")
        
        if len(spans) > 10:
            print(f"     ... and {len(spans) - 10} more spans")
            
    except Exception as e:
        logger.error(f"Failed to analyze trace file {trace_file}: {e}")


def merge_traces(trace_files: List[str], output_file: str) -> None:
    """Merge multiple trace files into one."""
    if not trace_files:
        print("No trace files specified for merging")
        return
    
    # Validate input files
    valid_files = []
    for trace_file in trace_files:
        if os.path.exists(trace_file):
            valid_files.append(trace_file)
        else:
            logger.warning(f"Trace file not found: {trace_file}")
    
    if not valid_files:
        print("No valid trace files found")
        return
    
    print(f"Merging {len(valid_files)} trace files into {output_file}...")
    
    if merge_trace_files(valid_files, output_file):
        print(f"✅ Successfully merged traces to {output_file}")
        analyze_trace(output_file)
    else:
        print("❌ Failed to merge trace files")


def export_config_template(output_file: str = "trace_config_template.json") -> None:
    """Export a template trace configuration file."""
    presets = create_trace_export_presets()
    
    template = {
        "description": "Trace export configuration template for tau-bench",
        "available_presets": list(presets.keys()),
        "preset_descriptions": {
            "phoenix_only": "Export traces only to Phoenix (default)",
            "file_only": "Export traces only to JSON files",
            "phoenix_and_file": "Export to both Phoenix and files",
            "full_export": "Export to Phoenix, files, and console",
            "otlp_collector": "Export to OTLP collector and files"
        },
        "custom_config_example": {
            "phoenix_enabled": True,
            "phoenix_port": 6006,
            "file_export_enabled": True,
            "file_export_path": "./traces",
            "file_export_format": "json",
            "console_export_enabled": False,
            "otlp_http_enabled": False,
            "otlp_http_endpoint": "http://localhost:4318/v1/traces",
            "otlp_grpc_enabled": False,
            "otlp_grpc_endpoint": "http://localhost:4317",
            "custom_attributes": {
                "service.name": "tau-bench",
                "service.version": "0.1.0"
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✅ Trace configuration template saved to {output_file}")


def convert_trace_format(input_file: str, output_file: str, target_format: str) -> None:
    """Convert trace file between different formats."""
    try:
        with open(input_file, 'r') as f:
            trace_data = json.load(f)
        
        if target_format == "otlp":
            # Convert to OTLP-like format
            converted = convert_to_otlp_format(trace_data)
        elif target_format == "jaeger":
            # Convert to Jaeger format (simplified)
            converted = convert_to_jaeger_format(trace_data)
        else:
            print(f"Unsupported target format: {target_format}")
            return
        
        with open(output_file, 'w') as f:
            json.dump(converted, f, indent=2, default=str)
        
        print(f"✅ Converted {input_file} to {target_format} format: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to convert trace format: {e}")


def convert_to_otlp_format(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tau-bench trace format to OTLP format."""
    spans = trace_data.get("spans", [])
    
    otlp_data = {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "tau-bench"}},
                    {"key": "service.version", "value": {"stringValue": "0.1.0"}}
                ]
            },
            "scopeSpans": [{
                "scope": {
                    "name": "tau_bench.agents.langgraph_tool_call_agent",
                    "version": "0.1.0"
                },
                "spans": [
                    {
                        "traceId": span.get("trace_id", ""),
                        "spanId": span.get("span_id", ""),
                        "parentSpanId": span.get("parent_span_id", ""),
                        "name": span.get("name", ""),
                        "startTimeUnixNano": str(span.get("start_time", 0)),
                        "endTimeUnixNano": str(span.get("end_time", 0)),
                        "attributes": [
                            {"key": k, "value": {"stringValue": str(v)}}
                            for k, v in span.get("attributes", {}).items()
                        ],
                        "status": {
                            "code": 1 if span.get("status", {}).get("status_code") == "OK" else 0,
                            "message": span.get("status", {}).get("description", "")
                        }
                    }
                    for span in spans
                ]
            }]
        }]
    }
    
    return otlp_data


def convert_to_jaeger_format(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tau-bench trace format to Jaeger format (simplified)."""
    spans = trace_data.get("spans", [])
    
    jaeger_spans = []
    for span in spans:
        jaeger_span = {
            "traceID": span.get("trace_id", ""),
            "spanID": span.get("span_id", ""),
            "parentSpanID": span.get("parent_span_id", ""),
            "operationName": span.get("name", ""),
            "startTime": span.get("start_time", 0),
            "duration": span.get("duration_ns", 0),
            "tags": [
                {"key": k, "value": str(v), "type": "string"}
                for k, v in span.get("attributes", {}).items()
            ],
            "process": {
                "serviceName": "tau-bench",
                "tags": [
                    {"key": "service.version", "value": "0.1.0", "type": "string"}
                ]
            }
        }
        jaeger_spans.append(jaeger_span)
    
    return {
        "data": [{
            "traceID": spans[0].get("trace_id", "") if spans else "",
            "spans": jaeger_spans,
            "processes": {
                "p1": {
                    "serviceName": "tau-bench",
                    "tags": [
                        {"key": "service.version", "value": "0.1.0", "type": "string"}
                    ]
                }
            }
        }]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Trace management utilities for tau-bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all trace files
  python trace_utils.py list

  # Analyze a specific trace
  python trace_utils.py analyze traces/trace_2024-01-01T10-00-00.json

  # Merge multiple traces
  python trace_utils.py merge traces/trace_*.json -o merged_traces.json

  # Export configuration template
  python trace_utils.py config-template

  # Convert trace format
  python trace_utils.py convert trace.json trace_otlp.json --format otlp
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all trace files')
    list_parser.add_argument('--dir', default='./traces', help='Traces directory (default: ./traces)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a trace file')
    analyze_parser.add_argument('trace_file', help='Path to trace file')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple trace files')
    merge_parser.add_argument('trace_files', nargs='+', help='Trace files to merge')
    merge_parser.add_argument('-o', '--output', required=True, help='Output file path')
    
    # Config template command
    config_parser = subparsers.add_parser('config-template', help='Export trace configuration template')
    config_parser.add_argument('-o', '--output', default='trace_config_template.json', help='Output file path')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert trace format')
    convert_parser.add_argument('input_file', help='Input trace file')
    convert_parser.add_argument('output_file', help='Output trace file')
    convert_parser.add_argument('--format', choices=['otlp', 'jaeger'], required=True, help='Target format')
    
    # Export from Phoenix command (placeholder)
    export_parser = subparsers.add_parser('export-phoenix', help='Export traces from Phoenix (experimental)')
    export_parser.add_argument('--port', type=int, default=6006, help='Phoenix port')
    export_parser.add_argument('-o', '--output', default='./exported_traces', help='Output directory')
    export_parser.add_argument('--format', choices=['json', 'otlp'], default='json', help='Export format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_traces(args.dir)
        elif args.command == 'analyze':
            analyze_trace(args.trace_file)
        elif args.command == 'merge':
            merge_traces(args.trace_files, args.output)
        elif args.command == 'config-template':
            export_config_template(args.output)
        elif args.command == 'convert':
            convert_trace_format(args.input_file, args.output_file, args.format)
        elif args.command == 'export-phoenix':
            result = export_traces_from_phoenix(args.port, args.output, args.format)
            if not result:
                print("Phoenix export is not yet implemented. Use file export during trace generation instead.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
