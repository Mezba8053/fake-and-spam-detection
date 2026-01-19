import json
import csv

input_file = "fake review/Cell_Phones_and_Accessories/Cell_Phones_and_Accessories.json"  
output_file = "output.csv"

with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", newline="", encoding="utf-8") as out_f:
    first_line = f.readline()
    first_obj = json.loads(first_line)

    fieldnames = first_obj.keys()
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(first_obj)
    count=50000
    for line in f:
        if line.strip() and count>0:  
            obj = json.loads(line)
            writer.writerow(obj)
            count -= 1

print("Conversion complete! Saved as output.csv")
