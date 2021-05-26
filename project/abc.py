# run black
from typing import NamedTuple
import statistics
import sqlite3
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename

# window = tk.Tk()

#### DATABASE
violations_schema = {
    "serial_number": "text",
    "violation_status": "text",
    "violation_code": "text",
    "violation_description": "text",
    "points": "integer",
}

inventory_schema = {
    "facility_id": "text",
    "facility_name": "text",
    "record_id": "text",
    "program_name": "text",
    "program_element": "integer",
    "pe_description": "text",
    "facility_address": "text",
    "facility_city": "text",
    "facility_state": "text",
    "facility_zip": "integer",
    "facility_latitude": "real",
    "facility_longitude": "real",
    "owner_id": "text",
    "owner_name": "text",
    "owner_address": "text",
    "owner_city": "text",
    "owner_state": "text",
    "owner_zip": "integer",
    "location": "text",  # get rid of this
    "census_tracts": "integer",
    "supervisorial_district_boundaries": "integer",
    "board_approved_statistical_areas": "integer",
    "zip_codes": "integer",
    "seating_type": "text",
    "seats": "text",
    "risk": "text",
}

inspections_schema = {
    "activity_date": "text",
    "owner_id": "text",
    "owner_name": "text",
    "facility_id": "text",
    "facility_name": "text",
    "record_id": "text",
    "program_name": "text",
    "program_status": "text",
    "program_element": "integer",
    "pe_description": "text",
    "facility_address": "text",
    "facility_city": "text",
    "facility_state": "text",
    "facility_zip": "integer",
    "service_code": "integer",
    "service_description": "text",
    "score": "integer",
    "grade": "text",
    "serial_number": "text",
    "employee_id": "text",
    "location": "text",
    "supervisorial_district_boundaries": "integer",
    "census_tracts": "integer",
    "board_approved_statistical_areas": "integer",
    "zip_codes": "integer",
    "seating_type": "text",
    "seats": "text",
    "risk": "text",
}

DB = "app.db"

def create_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    create_schema_string = lambda schema: ",".join(f"{k} {v}" for k, v in schema.items())
    c.execute(f"CREATE TABLE IF NOT EXISTS violations ({create_schema_string(violations_schema)})")
    c.execute(f"CREATE TABLE IF NOT EXISTS inventory ({create_schema_string(inventory_schema)})")
    c.execute(f"CREATE TABLE IF NOT EXISTS inspections ({create_schema_string(inspections_schema)})")
    conn.commit()
    conn.close()

def populate_db(table, df):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"Inserting row {i}...")
        c.execute(f"INSERT INTO {table} VALUES ({('?,'*len(df.columns))[:-1]});", row)
    conn.commit()
    conn.close()


def get_inspection_scores(year, *, seating_type=None, zip_code=None):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    if seating_type is not None and zip_code is not None:
        raise ValueError("Can't specify both seating_type and zip_code")
    rtn = None
    if seating_type is not None:
        c.execute("""
            SELECT score FROM inspections WHERE activity_date LIKE ? AND seating_type = ?
        """, (f"%/{year}", seating_type))
        rtn = c.fetchall()
    if zip_code is not None:
        c.execute("""
            SELECT score FROM inspections WHERE activity_date LIKE ? AND facility_zip = ?
        """, (f"%/{year}", zip_code))
        rtn = c.fetchall()
    conn.commit()
    conn.close()
    return [x[0] for x in rtn if isinstance(x[0], int)]


def get_all_violations():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT serial_number, violation_code FROM violations")
    violations = c.fetchall()
    conn.commit()
    conn.close()
    return violations


def serial_numbers_to_facility_id():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT serial_number, facility_id FROM inspections")
    rtn = dict(c.fetchall())
    conn.commit()
    conn.close()
    return rtn
####


def get_user_csv() -> pd.DataFrame:
    return pd.read_csv(askopenfilename())


def clean_and_insert_violations(df: pd.DataFrame):
    df.columns = ['serial_number', 'violation_status', 'violation_code', 'violation_description', 'points']
    def validate_row(row) -> bool:
        if len(row.serial_number) != 9 or not row.serial_number.startswith("DA"):
            return False
        if len(row.violation_code) != 4:
            return False
        if not isinstance(row.points, int):
            return False
        return True
    df.drop([i for i, row in df.iterrows() if not validate_row(row)])
    populate_db("violations", df)


def clean_and_insert(title: str, df: pd.DataFrame):
    df["type"] = df.apply(lambda x: x["PE DESCRIPTION"].split(" (")[0], axis=1)

    def extract_seats(s):
        split_str = s.split("(")
        if len(split_str) == 1:
            return None
        return split_str[1].split(")")[0].strip()
    df["seats"] = df.apply(lambda x: extract_seats(x["PE DESCRIPTION"]), axis=1)

    def extract_risk(s):
        if s.endswith("RISK"):
            return s.strip(" RISK").split()[-1]
        return None
    df["risk"] = df.apply(lambda x: extract_risk(x["PE DESCRIPTION"]), axis=1)
    populate_db(title, df)


# create_db()
# clean_and_insert('inventory', pd.read_csv("inputs/inventory.csv"))
# clean_and_insert('inspections', pd.read_csv("inputs/inspections.csv"))
# window.mainloop()


class InspectionStats(NamedTuple):
    year: int
    seating_type: str
    zip_code: int
    count: int
    mean: int
    median: int
    mode: int


def get_inspection_stats(year, seating_type=None, zip_code=None):
    scores = get_inspection_scores(year, seating_type=seating_type, zip_code=zip_code)
    return InspectionStats(
        year=year,
        seating_type=seating_type,
        zip_code=zip_code,
        count=len(scores),
        mean=statistics.mean(scores),
        median=statistics.median(scores),
        mode=statistics.mode(scores),
    )


def graph_violations():
    data = get_all_violations()

    ###
    facility_id_lookup = serial_numbers_to_facility_id()
    all_serials_I = set(facility_id_lookup.keys())
    all_serials_V = {x[0] for x in data}
    problems = all_serials_V - all_serials_I
    ###

    collated_violations = {}
    for serial_number, violation_code in data:
        if violation_code in collated_violations:
            collated_violations[violation_code].append(serial_number)
        else:
            collated_violations[violation_code] = [serial_number]

    for k, v in collated_violations.items():
        collated_violations[k] = len({facility_id_lookup[serial_number] for serial_number in v if serial_number not in problems})
    collated_violations['Other'] = 0
    keys_to_delete = []
    for k, v in collated_violations.items():
        if v < 2000:
            collated_violations['Other'] += v
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del collated_violations[k]
    print(collated_violations)    
    
    x = list(collated_violations.keys())
    y = list(collated_violations.values())
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    sns.barplot(x=x, y=y)
    plt.show()


def print_violation_descriptions():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT violation_code, violation_description FROM violations")
    descs = c.fetchall()
    conn.commit()
    conn.close()
    a = {k: v for k, v in descs}
    for k, v in a.items():
        print(f"{k}: {v}")

graph_violations()
