"""
Exploratory Data Analysis of OASIS-1 and OASIS-2 datasets
Shauna Vronces
June 24, 2026
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

def load_data(filename):
    df = pd.read_excel(filename)
    return df

def clean_data(data, data_type):
    educ_map = {
        1: "less than high school grad",
        2: "high school grad",
        3: "some college",
        4: "college grad",
        5: "beyond college"
    }

    if data_type == 1:
        cols_to_keep = ["ID", 
                    "M/F", 
                    "Age",
                    "Educ", 
                    "SES",
                    "MMSE", 
                    "CDR"]
        
        data = data[cols_to_keep].copy()
        data["Educ"] = data["Educ"].map(educ_map)
    
    elif data_type == 2:
        cols_to_keep = ["Subject ID", 
                    "MRI ID", 
                    "Group", 
                    "Visit", 
                    "M/F", 
                    "Age", 
                    "EDUC", 
                    "SES", 
                    "MMSE", 
                    "CDR"]
        data = data[cols_to_keep].copy()
    else:
        raise ValueError("data_type must be 1 (OASIS-1) or 2 (OASIS-2)")
    
    return data

def handle_missing(data):
    print(f"Missing values: {data.isna().sum()}")
    return data

def get_summary(data, data_type):
    if data_type == 1:
        data.info()

        print("\nDescriptive statistics:")
        print(data.describe(include="all"))

        print("\nCDR counts:")
        print(data["CDR"].value_counts(dropna=False))

        print("\nSex counts:")
        print(data["M/F"].value_counts(dropna=False))

        print("\nSES counts:")
        print(data["SES"].value_counts(dropna=False))

        print("\nEducation counts:")
        print(data["Educ"].value_counts(dropna=False))

    elif data_type == 2:
        data = data[data["Visit"] == 1].copy()

        data.info()

        print("\nDescriptive statistics:")
        print(data.describe(include="all"))

        print("\nCDR counts:")
        print(data["CDR"].value_counts(dropna=False))

        print("\nSex counts:")
        print(data["M/F"].value_counts(dropna=False))

        print("\nSES counts:")
        print(data["SES"].value_counts(dropna=False))

        print("\nEducation (years spent in school) counts:")
        print(data["EDUC"].value_counts(dropna=False))

    else:
        raise ValueError("data_type must be 1 or 2")
    
def eda(data, data_type):
    if data_type == 1:
        # Age dist
        plt.figure(figsize=(8,6))
        plt.hist(data["Age"], bins=15, edgecolor="black")
        plt.title(f"Oasis-1: Age Distribution")
        plt.xlabel("Age (years)")
        plt.ylabel("Number of Subjects")
        plt.tight_layout()
        plt.savefig("oasis1_age_dist.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Sex dist
        plt.figure(figsize=(8,6))
        data["M/F"].value_counts().plot(kind="bar", color=["#d1553e", "#4d88b9"])
        plt.title(f"Oasis-1: Sex Distribution")
        plt.xlabel("Sex")
        plt.xticks(rotation=0)
        plt.ylabel("Number of Subjects")
        plt.tight_layout()
        plt.savefig("oasis1_sex_dist.png", dpi=300, bbox_inches="tight")
        plt.close()

        # CDR Dist
        # Dropping NaN values in CDR for Oasis-1, because almost all of these NaN cases come
        # individuals under the age of 30, most likely control group
        plt.figure(figsize=(8,6))
        data["CDR"].value_counts(dropna=True).sort_index().plot(kind="bar")
        plt.title("Oasis-1: CDR Distribution")
        plt.xlabel("Clinical Dementia Rating (CDR)")
        plt.ylabel("Number of Subjects")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig("oasis1_cdr_dist.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Age by CDR
        plt.figure(figsize=(8,6))
        data.dropna(subset=["CDR"]).boxplot(column="Age", by="CDR")
        plt.suptitle("")
        plt.title("OASIS-1: Age by CDR")
        plt.xlabel("Clinical Dementia Rating (CDR)")
        plt.ylabel("Age (years)")
        plt.tight_layout()
        plt.savefig("oasis1_age_by_cdr.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Sex by CDR
        plt.figure(figsize=(8,6))
        sex_cdr = pd.crosstab(data["CDR"], data["M/F"])
        sex_cdr.plot(kind="bar", stacked=True, color=["#d1553e", "#4d88b9"])
        plt.title("OASIS-1: Sex by CDR")
        plt.xlabel("Clinical Dementia Rating (CDR)")
        plt.ylabel("Number of Subjects")
        plt.xticks(rotation=0)
        plt.legend(title="Sex")
        plt.tight_layout()
        plt.savefig("oasis1_sex_by_cdr.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # MMSE Dist
        plt.figure(figsize=(8,6))
        plt.hist(data["MMSE"].dropna(), bins=15, edgecolor="black")
        plt.title("Oasis-1: MMSE Distribution")
        plt.xlabel("Mini-Mental State Examination (MMSE)")
        plt.ylabel("Number of Subjects")
        plt.tight_layout()
        plt.savefig("oasis1_mmse_dist.png", dpi=300, bbox_inches="tight")
        plt.close()

        # MMSE by CDR
        plt.figure(figsize=(8,6))
        data.dropna(subset=["CDR","MMSE"]).boxplot(column="MMSE", by="CDR")
        plt.suptitle("")
        plt.title("OASIS-1: MMSE by CDR")
        plt.xlabel("Clinical Dementia Rating (CDR)")
        plt.ylabel("MMSE Score")
        plt.tight_layout()
        plt.savefig("oasis1_mmse_by_cdr.png", dpi=300, bbox_inches="tight")
        plt.close()

    elif data_type == 2:
        data = data[data["Visit"] == 1].copy()

        # Age dist
        plt.figure(figsize=(8,6))
        plt.hist(data["Age"], bins=15, edgecolor="black")
        plt.title(f"Oasis-2: Age Distribution at First Visit")
        plt.xlabel("Age (years)")
        plt.ylabel("Number of Subjects")
        plt.tight_layout()
        plt.savefig("oasis2_age_dist.png", dpi=300, bbox_inches="tight")
        plt.close()

         # Sex dist
        plt.figure(figsize=(8,6))
        data["M/F"].value_counts().plot(kind="bar", color=["#d1553e", "#4d88b9"])
        plt.title(f"Oasis-2: Sex Distribution")
        plt.xlabel("Sex")
        plt.ylabel("Number of Subjects")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig("oasis2_sex_dist.png", dpi=300, bbox_inches="tight")
        plt.close()

        # CDR Dist
        plt.figure(figsize=(8,6))
        data["CDR"].value_counts(dropna=False).sort_index().plot(kind="bar")
        plt.title("Oasis-2: CDR Distribution")
        plt.xlabel("Clinical Dementia Rating (CDR) at First Visit")
        plt.ylabel("Number of Subjects")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig("oasis2_cdr_dist.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Age by CDR
        plt.figure(figsize=(8,6))
        data.dropna(subset=["CDR"]).boxplot(column="Age", by="CDR")
        plt.suptitle("")
        plt.title("OASIS-2: Age by CDR at First Visit")
        plt.xlabel("Clinical Dementia Rating (CDR)")
        plt.ylabel("Age (years)")
        plt.tight_layout()
        plt.savefig("oasis2_age_by_cdr.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Sex by CDR
        plt.figure(figsize=(8,6))
        sex_cdr = pd.crosstab(data["CDR"], data["M/F"])
        sex_cdr.plot(kind="bar", stacked=True, color=["#d1553e", "#4d88b9"])
        plt.title("OASIS-2: Sex by CDR")
        plt.xlabel("Clinical Dementia Rating (CDR)")
        plt.ylabel("Number of Subjects")
        plt.xticks(rotation=0)
        plt.legend(title="Sex")
        plt.tight_layout()
        plt.savefig("oasis2_sex_by_cdr.png", dpi=300, bbox_inches="tight")
        plt.close()

        # MMSE Dist
        plt.figure(figsize=(8,6))
        plt.hist(data["MMSE"].dropna(), bins=15, edgecolor="black")
        plt.title("Oasis-2: MMSE Distribution")
        plt.xlabel("Mini-Mental State Examination (MMSE) at First Visit")
        plt.ylabel("Number of Subjects")
        plt.tight_layout()
        plt.savefig("oasis2_mmse_dist.png", dpi=300, bbox_inches="tight")
        plt.close()

        # MMSE by CDR
        plt.figure(figsize=(8,6))
        data.dropna(subset=["CDR","MMSE"]).boxplot(column="MMSE", by="CDR")
        plt.suptitle("")
        plt.title("OASIS-2: MMSE by CDR at First Visit")
        plt.xlabel("Clinical Dementia Rating (CDR)")
        plt.ylabel("MMSE Score")
        plt.tight_layout()
        plt.savefig("oasis2_mmse_by_cdr.png", dpi=300, bbox_inches="tight")
        plt.close()

    else:
        raise ValueError("data_type must be 1 (OASIS-1) or 2 (OASIS-2)")


if __name__ == "__main__":
    oasis1 = load_data("oasis1_cross-sectional-5708aa0a98d82080.xlsx")
    oasis2 = load_data("oasis2_longitudinal_demographics-8d83e569fa2e2d30.xlsx")

    cleaned_oasis1 = handle_missing(clean_data(oasis1, 1))
    cleaned_oasis2 = handle_missing(clean_data(oasis2, 2))

    print("================================== \n OASIS-1 Summary")
    get_summary(cleaned_oasis1, 1)

    print("================================== \n EDA Oasis-1: Generating and saving plots")
    eda(cleaned_oasis1, 1)

    print("\n \n ================================== \n OASIS-2 Summary")
    get_summary(cleaned_oasis2, 2)

    print("================================== \n EDA Oasis-2: Generating and saving plots" )
    eda(cleaned_oasis2, 2)





