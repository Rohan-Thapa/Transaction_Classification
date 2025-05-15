import csv
import re

# Paths of the all required files.
input_log = 'digital_wallet.log' # this is through the DigiPocket Project of Week3, the logs generated through Observer Design pattern.
output_csv = 'transactions_datasets.csv'

# Regex to parse lines from the original simple log format got the regex through the internet as writing regex oneself if the gigachad.
pattern = re.compile(
    r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - TX (?P<tx_id>[0-9a-f]+) \|"
    r" (?P<sender>[0-9a-f]+)->(?P<recipient_type>\w+):(?P<recipient_id>[0-9a-f]+) \|"
    r" (?P<amount>\d+\.?\d*) \| (?P<description>.+)"
)

# Convert log to CSV as I'll be using CSV format for the datasets.
def convert_log_to_csv(log_path, csv_path):
    with open(log_path, 'r') as logfile, open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                'datetime', 'tx_id', 'sender',
                'recipient_type', 'recipient_id', 'amount', 'description'
            ]
        )
        writer.writeheader()

        for line in logfile:
            match = pattern.match(line.strip())
            if match:
                row = match.groupdict()
                writer.writerow(row)

    print(f"Converted '{log_path}' to '{csv_path}' successfully.")

if __name__ == '__main__':
    convert_log_to_csv(input_log, output_csv)
