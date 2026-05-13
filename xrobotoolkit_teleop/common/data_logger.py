import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

try:
    from scripts.misc.check_teleop_log_health import check_log, print_report
except ImportError:  # pragma: no cover - optional repo-local helper
    check_log = None
    print_report = None


class DataLogger:
    """
    A simple data logger that collects data entries and saves them to a pickle file.
    """

    def __init__(self, log_dir="logs", validate_before_save=False, decode_images_on_validate=True):
        """
        Initializes the logger.

        Args:
            log_dir (str): The directory where log files will be stored. If None, logging is disabled.
            validate_before_save (bool): Run log health checks before promoting a log file.
            decode_images_on_validate (bool): Decode image payloads during validation.
        """
        self.log_data = []
        self.log_dir = log_dir
        self.validate_before_save = bool(validate_before_save)
        self.decode_images_on_validate = bool(decode_images_on_validate)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.count = 0
        self.log_file = None
        os.makedirs(log_dir, exist_ok=True)

    def _health_check_args(self):
        return SimpleNamespace(
            required_keys=None,
            min_entries=2,
            min_hz=1.0,
            max_hz=120.0,
            max_dt=0.5,
            joint_limit=2 * 3.141592653589793,
            velocity_limit=20.0,
            max_jump=1.0,
            static_action_travel=0.5,
            static_state_travel=0.5,
            static_command_range=0.2,
            static_state_range=0.2,
            decode_images=self.decode_images_on_validate,
            strict_warnings=False,
            max_issues=20,
        )

    def _validate_log_file(self, path):
        if check_log is None:
            print("Warning: log health checker is unavailable; skipping validation.")
            return True

        args = self._health_check_args()
        report = check_log(Path(path), args)
        if print_report is not None:
            print_report(report, args.max_issues)
        if report.error_count:
            print(f"Log health check failed with {report.error_count} error(s).")
            return False
        return True

    def add_entry(self, data_entry):
        """
        Adds a data entry to the log. A timestamp is automatically added.

        Args:
            data_entry (dict): A dictionary containing the data to log for the current timestep.
        """
        self.log_data.append(data_entry)

    def save(self):
        """
        Saves the collected log data to a pickle file.
        """
        if not self.log_data:
            print("No data to save.")
            return
        self.count += 1
        final_log_file = os.path.join(self.log_dir, f"teleop_log_{self.timestamp}_{self.count}.pkl")

        print(f"Saving {len(self.log_data)} data points to {final_log_file}...")
        tmp_file = None
        try:
            fd, tmp_file = tempfile.mkstemp(
                prefix=f".teleop_log_{self.timestamp}_{self.count}_",
                suffix=".tmp.pkl",
                dir=self.log_dir,
            )
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self.log_data, f)
            if self.validate_before_save and not self._validate_log_file(tmp_file):
                os.remove(tmp_file)
                self.log_file = None
                print("Rejected unhealthy log; no final log file was saved.")
                return
            os.replace(tmp_file, final_log_file)
            self.log_file = final_log_file
            print(f"Data successfully saved to {self.log_file}")
        except OSError as e:
            print(f"Error saving data: {e}")
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

    def reset(self):
        """
        Resets the logger, clearing all collected data.
        """
        self.log_data = []
        self.log_file = None

        print("Logger has been reset.")
