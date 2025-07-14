#!/usr/bin/env python3
"""
Grafana Dashboard Manager - Export/Import dashboards easily.

This script helps manage Grafana dashboards following KISS principles.
"""

import json
import os
import sys
import requests
from typing import Dict, List, Optional
import argparse


class GrafanaDashboardManager:
    def __init__(self, grafana_url: str = "http://localhost:3000", 
                 username: str = "admin", password: str = "admin"):
        self.grafana_url = grafana_url.rstrip('/')
        self.auth = (username, password)
        self.headers = {"Content-Type": "application/json"}
    
    def export_dashboard(self, uid: str, output_file: str) -> bool:
        """Export a dashboard by UID to a JSON file."""
        try:
            # Get dashboard
            response = requests.get(
                f"{self.grafana_url}/api/dashboards/uid/{uid}",
                auth=self.auth
            )
            response.raise_for_status()
            
            dashboard_data = response.json()
            dashboard = dashboard_data.get('dashboard', {})
            
            # Clean up for export
            dashboard.pop('id', None)
            dashboard.pop('uid', None)
            dashboard.pop('version', None)
            dashboard['__inputs'] = []
            dashboard['__requires'] = []
            
            # Save to file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(dashboard, f, indent=2)
            
            print(f"✓ Exported dashboard '{dashboard.get('title')}' to {output_file}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to export dashboard: {e}")
            return False
    
    def import_dashboard(self, input_file: str, folder_id: int = 0) -> bool:
        """Import a dashboard from a JSON file."""
        try:
            # Load dashboard
            with open(input_file, 'r') as f:
                dashboard = json.load(f)
            
            # Prepare import payload
            payload = {
                "dashboard": dashboard,
                "folderId": folder_id,
                "overwrite": True,
                "inputs": []
            }
            
            # Import dashboard
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                auth=self.auth,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"✓ Imported dashboard '{dashboard.get('title')}' (UID: {result.get('uid')})")
            return True
            
        except (requests.exceptions.RequestException, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"✗ Failed to import dashboard: {e}")
            return False
    
    def list_dashboards(self) -> List[Dict]:
        """List all dashboards."""
        try:
            response = requests.get(
                f"{self.grafana_url}/api/search?type=dash-db",
                auth=self.auth
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return []
    
    def export_all_dashboards(self, output_dir: str = "dashboards") -> int:
        """Export all dashboards to a directory."""
        dashboards = self.list_dashboards()
        if not dashboards:
            print("No dashboards found.")
            return 0
        
        exported = 0
        for dash in dashboards:
            uid = dash.get('uid')
            title = dash.get('title', 'untitled').replace('/', '-')
            output_file = os.path.join(output_dir, f"{title}.json")
            
            if self.export_dashboard(uid, output_file):
                exported += 1
        
        print(f"\nExported {exported}/{len(dashboards)} dashboards to {output_dir}/")
        return exported


def main():
    parser = argparse.ArgumentParser(
        description="Manage Grafana dashboards - export/import made easy"
    )
    parser.add_argument(
        "--url", 
        default="http://localhost:3000",
        help="Grafana URL (default: http://localhost:3000)"
    )
    parser.add_argument(
        "--username", 
        default="admin",
        help="Grafana username (default: admin)"
    )
    parser.add_argument(
        "--password", 
        default="admin",
        help="Grafana password (default: admin)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export dashboard(s)")
    export_parser.add_argument("--uid", help="Dashboard UID to export")
    export_parser.add_argument("--all", action="store_true", help="Export all dashboards")
    export_parser.add_argument(
        "--output", "-o",
        default="dashboards",
        help="Output file or directory (default: dashboards)"
    )
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import dashboard")
    import_parser.add_argument("file", help="Dashboard JSON file to import")
    import_parser.add_argument(
        "--folder-id", 
        type=int, 
        default=0,
        help="Folder ID to import into (default: 0 - General)"
    )
    
    # List command
    subparsers.add_parser("list", help="List all dashboards")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize manager
    manager = GrafanaDashboardManager(
        grafana_url=args.url,
        username=args.username,
        password=args.password
    )
    
    # Execute command
    if args.command == "export":
        if args.all:
            manager.export_all_dashboards(args.output)
        elif args.uid:
            output_file = args.output
            if os.path.isdir(output_file):
                output_file = os.path.join(output_file, f"{args.uid}.json")
            manager.export_dashboard(args.uid, output_file)
        else:
            print("Error: Specify --uid or --all for export")
            sys.exit(1)
    
    elif args.command == "import":
        manager.import_dashboard(args.file, args.folder_id)
    
    elif args.command == "list":
        dashboards = manager.list_dashboards()
        if dashboards:
            print("\nAvailable dashboards:")
            print("-" * 50)
            for dash in dashboards:
                print(f"UID: {dash.get('uid'):20} Title: {dash.get('title')}")
        else:
            print("No dashboards found.")


if __name__ == "__main__":
    main() 