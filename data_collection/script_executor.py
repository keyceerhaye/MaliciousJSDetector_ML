import json
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class ScriptExecutor:
    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None
        self.logs = {
            'dom_changes': [],
            'js_api_calls': [],
            'network_requests': [],
            'alerts': [],
            'cpu_usage': []
        }

    def setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        # Enable performance logging
        chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL", "browser": "ALL"})
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_cdp_cmd('Network.enable', {})
        self.driver.execute_cdp_cmd('Performance.enable', {})
        self.driver.execute_cdp_cmd('DOM.enable', {})
        self.driver.execute_cdp_cmd('Console.enable', {})
        self.driver.execute_cdp_cmd('Page.enable', {})
        self.driver.execute_cdp_cmd('Runtime.enable', {})

    def inject_logging_script(self):
        logging_script = """
        // Override eval to log calls
        const originalEval = window.eval;
        window.eval = function(code) {
            console.log('Eval called with:', code);
            return originalEval.apply(this, arguments);
        };

        // Override alert to log calls
        const originalAlert = window.alert;
        window.alert = function(message) {
            console.log('Alert called with:', message);
            return originalAlert.apply(this, arguments);
        };

        // Monitor DOM changes
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                console.log('DOM change:', mutation.type, mutation.target);
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
        """
        self.driver.execute_script(logging_script)

    def collect_logs(self):
        # Collect console logs
        console_logs = self.driver.get_log('browser')
        for log in console_logs:
            if 'Eval called with:' in log['message']:
                self.logs['js_api_calls'].append(log['message'])
            elif 'Alert called with:' in log['message']:
                self.logs['alerts'].append(log['message'])
            elif 'DOM change:' in log['message']:
                self.logs['dom_changes'].append(log['message'])

        # Collect network requests from performance logs
        performance_logs = self.driver.get_log("performance")
        for entry in performance_logs:
            try:
                log = json.loads(entry["message"])["message"]
                if log["method"] == "Network.requestWillBeSent":
                    self.logs["network_requests"].append(log["params"])
            except (KeyError, json.JSONDecodeError):
                continue

        # Collect performance metrics
        try:
            performance_metrics = self.driver.execute_cdp_cmd('Performance.getMetrics', {})
            if performance_metrics:
                self.logs['cpu_usage'].append(performance_metrics)
        except Exception as e:
            print(f"Warning: Could not collect performance metrics: {str(e)}")

    def execute_script(self, script_path, wait_time=10):
        try:
            self.setup_driver()
            self.inject_logging_script()
            
            # Create a simple HTML page that loads the script
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Script Test</title>
            </head>
            <body>
                <script src="{script_path}"></script>
            </body>
            </html>
            """
            
            # Save the HTML content to a temporary file
            html_path = 'temp_test.html'
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            # Load the HTML file
            self.driver.get(f'file:///{os.path.abspath(html_path)}')
            time.sleep(wait_time)
            self.collect_logs()
            
            # Clean up the temporary HTML file
            os.remove(html_path)
            
        finally:
            if self.driver:
                self.driver.quit()

    def save_logs(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.logs, f, indent=4)

if __name__ == '__main__':
    try:
        executor = ScriptExecutor(headless=True)
        # Get the absolute path of the test script
        script_path = os.path.abspath('test_script.js')
        print(f"Executing script: {script_path}")
        executor.execute_script(script_path)
        executor.save_logs('behavior_logs.json')
        print("Script execution completed. Logs saved to behavior_logs.json")
    except Exception as e:
        print(f"An error occurred: {str(e)}") 