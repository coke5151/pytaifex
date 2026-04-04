import contextlib
import os
import time

import psutil
from pywinauto import Application, ElementNotFoundError
from pywinauto.timings import TimeoutError as PywinautoTimeoutError


class OfficialTTB:
    def __init__(self, ttb_path: str, force: bool = False):
        self._abs_ttb_path = os.path.abspath(ttb_path)

        existing_procs = []
        for p in psutil.process_iter():
            try:
                exe = p.exe()
                if exe and os.path.abspath(exe) == self._abs_ttb_path:
                    existing_procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if existing_procs:
            if force:
                for p in existing_procs:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        p.kill()
                psutil.wait_procs(existing_procs, timeout=3)
            else:
                raise RuntimeError(f"TTB is already running! (Path: {self._abs_ttb_path})")

        self.app = Application(backend="uia").start(ttb_path)
        self.win = win = self.app.window(title_re=r".*TTB-Ver:.*")
        win.wait("exists visible enabled ready", timeout=30)

    def __del__(self):
        self.app.kill()

    def login(self, account: str, password: str):
        self.win.set_focus()

        # 1. find account input box
        acc_edit = self.win.child_window(auto_id="txtAccnt", control_type="Edit")
        acc_edit.wait("exists visible enabled ready", timeout=10)

        ## clear and input account
        acc_edit.click_input()
        acc_edit.type_keys("^a{BACKSPACE}")  # Ctrl+A and Delete
        time.sleep(0.2)
        acc_edit.type_keys(account, with_spaces=True)

        # 2. find password input box
        pwd_edit = self.win.child_window(auto_id="txtPwd", control_type="Edit")
        pwd_edit.wait("exists visible enabled ready", timeout=10)

        ## clear and input password
        pwd_edit.click_input()
        pwd_edit.type_keys("^a{BACKSPACE}")
        time.sleep(0.2)
        pwd_edit.type_keys(password, with_spaces=True)

        # click login
        login_btn = self.win.child_window(title="登入", control_type="Button")
        login_btn.wait("exists visible enabled ready", timeout=10)
        login_btn.click_input()
        self.check_error_window()

    def check_error_window(self):
        time.sleep(2)

        # Detect error window under the main window
        try:
            error_win = self.win.child_window(title_re=r".*登入失敗.*|.*錯誤.*", control_type="Window")
            if error_win.exists():
                error_text = self.extract_error_content(error_win)

                # Close error window
                try:
                    ok_btn = error_win.child_window(title="確定", control_type="Button")
                    ok_btn.click_input()
                except ElementNotFoundError:
                    # if "確定" do not exist, try to click "X"
                    close_btn = error_win.child_window(control_type="Button", found_index=0)
                    close_btn.click_input()

                raise RuntimeError(f"TTB login failed: {error_text}")
        except PywinautoTimeoutError:
            pass
        except ElementNotFoundError:
            pass

    def extract_error_content(self, error_win):
        """Extract full content from the error window."""
        content_parts = []

        # 1. Window title
        content_parts.append(f"Title: {error_win.window_text()}")

        # 2. Fetch all Text/Static controls
        for ctrl_type in ["Text", "Static"]:
            texts = error_win.descendants(control_type=ctrl_type)  # Single type

            for text_ctrl in texts:
                text = text_ctrl.window_text().strip()
                if text and len(text) > 3:  # Filter out empty strings and texts that are too short
                    content_parts.append(f"Text: {text}")

            # 3. Fetch Document/RichText
            try:
                doc = error_win.child_window(control_type="Document")
                content_parts.append(f"Document: {doc.window_text().strip()}")
            except Exception:
                pass

            # 4. Fetch Edit controls (plain text)
            try:
                edits = error_win.descendants(control_type="Edit")
                for edit in edits:
                    content_parts.append(f"Edit: {edit.window_text().strip()}")
            except Exception:
                pass

            # 5. Legacy property fallback
            try:
                legacy_text = error_win.legacy_properties().get("Text", "")
                if legacy_text.strip():
                    content_parts.append(f"Legacy: {legacy_text.strip()}")
            except Exception:
                pass

            return " | ".join(content_parts)
