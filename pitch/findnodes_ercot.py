"""
Comanche Peak Node Finder
--------------------------
The plant does NOT appear as CPNPP or COMANCHE in ERCOT's settlement
point list because ERCOT names resource nodes after the electrical bus
on the transmission network, not the plant.

From the NRC interconnection filing, Comanche Peak's three
interconnecting lines terminate at:
  1.  PARKER 345kV switching station  (most likely the primary RN bus)
  2.  COMANCHE SWITCH 345kV
  3.  STEPHENVILLE 138kV  (smaller, less relevant)

So the ERCOT resource node name will contain one of:
  PARKER, COMANCHE_SW, COMANCH, or a Luminant/Vistra prefix like LUM_.

Run the cells below in order.
"""

import pandas as pd
import gridstatus
from gridstatus.ercot import SETTLEMENT_POINTS_LIST_AND_ELECTRICAL_BUSES_MAPPING_RTID


def get_avail_methods_ercot():

    iso = gridstatus.Ercot()

    # See everything available
    print([a for a in dir(iso) if not a.startswith("__")])


def get_candidate_bus():
    iso = gridstatus.Ercot()
    # doc = iso._get_document(
    #     report_type_id=SETTLEMENT_POINTS_LIST_AND_ELECTRICAL_BUSES_MAPPING_RTID,
    #     verbose=True,
    # )
    # sp_list = pd.read_csv(gridstatus.utils.get_zip_file(doc.url))
    # print(sp_list.columns.tolist())

    # search_terms = [
    #     "PARKER",
    #     "COMANCH",
    #     "COMANCHE_SW",
    #     "LUM_",
    #     "LUMINANT",
    #     "VISTRA",
    #     "STEPHENVILLE",
    #     "GLEN",
    #     "SOMERVELL",
    #     "NUCLEAR",
    # ]
    # mask = sp_list.apply(
    #     lambda col: col.astype(str).str.contains("|".join(search_terms), case=False)
    # ).any(axis=1)
    # print(sp_list[mask])
    buses = iso.get_lmp(
        date="today",
        location_type="electrical bus",  # <-- this is the key change
    )

    all_locations = (
        buses[["Location", "Location Type"]]
        .drop_duplicates()
        .sort_values("Location")
        .reset_index(drop=True)
    )

    print(f"Total electrical buses: {len(all_locations)}")

    # Now search — the bus will be named after the Oncor substation.
    # From the NRC interconnection filing the 345kV lines go to:
    #   PARKER switching station and COMANCHE SWITCH
    # Luminant may also have registered under an internal plant code.

    search_terms = ["PARKER", "COMANCH", "LUM_", "CPNP", "GLEN", "SOMERVELL", "NUCLEAR"]
    mask = all_locations["Location"].str.contains(
        "|".join(search_terms), case=False, na=False
    )
    print("\nMatches (Bus):")
    candidate_buses = all_locations[mask]
    return candidate_buses


def get_node_name_per_bus_list(candidate_buses):
    iso = gridstatus.Ercot()

    mapping = iso._get_settlement_point_mapping()
    # Find the bus and settlement point column names from whatever it returns
    bus_col = "ELECTRICAL_BUS"  # 'PSSE_BUS_NAME' #next(c for c in mapping.columns if "bus" in c.lower())
    # sp_col = next(
    #     c for c in mapping.columns if "settlement" in c.lower() or "point" in c.lower()
    # )

    print(f"\nUsing columns: bus='{bus_col}', settlement_point='{sp_col}'")

    result = mapping[mapping[bus_col].isin(candidate_buses)]
    print(result[[sp_col, bus_col]].to_string(index=False))
    return result


if __name__ == "__main__":

    candidate_buses = get_candidate_bus()

    print(candidate_buses.to_string(index=False))

    nodes = get_node_name_per_bus_list(candidate_buses)

    # iso = gridstatus.Ercot()

    # # The settlement point <-> electrical bus mapping table
    # # This is the NP4-160-SG report -- the same one we tried earlier
    # # but now accessed through the API parser constants

    # lmp = iso.get_lmp(
    #     date="2021-02-15",          # Peak of Uri -- will show extreme prices
    #     end="2021-02-16",
    #     location_type="electrical bus",
    # )

    # # print(mapping.columns.tolist())
    # # bus_col = next(c for c in mapping.columns if "bus" in c.lower())
    # # sp_col  = next(c for c in mapping.columns if "settlement" in c.lower() or "point" in c.lower())

    # # mask = mapping[bus_col].isin(candidate_buses)
    # # print(mapping[mask][[sp_col, bus_col]].to_string(index=False))
