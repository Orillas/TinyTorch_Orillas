# tito/commands/login.py
import time
import json
import urllib.parse
from pathlib import Path
from argparse import ArgumentParser, Namespace
from rich.prompt import Confirm
from tito.commands.base import BaseCommand
from tito.core.auth import AuthReceiver, save_credentials, delete_credentials, ENDPOINTS, is_logged_in
from tito.core.browser import open_url

class LoginCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "login"

    @property
    def description(self) -> str:
        return "Log in to TinyTorch via web browser"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--force", action="store_true", help="Force re-login")

    def run(self, args: Namespace) -> int:
        # Adapted logic from api.py
        if args.force:
            delete_credentials()
            self.console.print("Cleared existing credentials.")

        # Check if already logged in (unless force was used)
        if is_logged_in():
            self.console.print("[green]You are already logged in.[/green]")
            if Confirm.ask("[bold yellow]Do you want to force re-login?[/bold yellow]", default=False):
                delete_credentials()
                self.console.print("Cleared existing credentials. Proceeding with new login...")
            else:
                self.console.print("Login cancelled.")
                return 0

        receiver = AuthReceiver()
        try:
            port = receiver.start()

            # Construct URL with optional profile pre-fill
            params = {"redirect_port": str(port)}
            
            # Try to read local profile for auto-fill
            try:
                # Check project root first, then global home
                project_profile = Path("profile.json").resolve()
                global_profile = Path.home() / ".tinytorch" / "profile.json"
                
                profile_path = None
                if project_profile.exists():
                    profile_path = project_profile
                elif global_profile.exists():
                    profile_path = global_profile

                if profile_path:
                    self.console.print(f"[dim]Reading profile from: {profile_path}[/dim]")
                    with open(profile_path, 'r') as f:
                        profile = json.load(f)
                        self.console.print(f"[dim]Found profile data: {profile.get('email', 'No email')}[/dim]")
                        if "email" in profile: params["email"] = profile["email"]
                        if "name" in profile: params["name"] = profile["name"]
                        if "affiliation" in profile: params["affiliation"] = profile["affiliation"]
                else:
                    self.console.print(f"[yellow]No profile found (checked ./profile.json and {global_profile})[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Failed to read profile: {e}[/red]")


            query_string = urllib.parse.urlencode(params)
            target_url = f"{ENDPOINTS['cli_login']}?{query_string}"

            # Use cross-platform browser opener
            open_url(target_url, self.console, show_manual_fallback=True)
            self.console.print("\n[dim]Waiting for authentication...[/dim]")
            
            tokens = receiver.wait_for_tokens()
            if tokens:
                save_credentials(tokens)
                self.console.print(f"[green]Success! Logged in as {tokens['user_email']}[/green]")
                return 0
            else:
                self.console.print("[red]Login timed out.[/red]")
                return 1
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            return 1


class LogoutCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "logout"

    @property
    def description(self) -> str:
        return "Log out of TinyTorch by clearing stored credentials"

    def add_arguments(self, parser: ArgumentParser) -> None:
        pass  # No arguments needed

    def run(self, args: Namespace) -> int:
        try:
            # Start local server for logout redirect
            receiver = AuthReceiver()
            port = receiver.start()

            # Open browser to local logout endpoint
            logout_url = f"http://127.0.0.1:{port}/logout"
            self.console.print(f"Opening browser to complete logout...")
            open_url(logout_url, self.console, show_manual_fallback=False)

            # Give browser time to redirect and close
            time.sleep(2.0)

            # Clean up server
            receiver.stop()

            # Delete local credentials
            delete_credentials()
            self.console.print("[green]Successfully logged out of TinyTorch![/green]")
            return 0
        except Exception as e:
            self.console.print(f"[red]Error during logout: {e}[/red]")
            return 1
