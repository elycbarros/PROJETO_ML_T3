"""Engenharia vigente com validação dos operandos antes da divisão."""
from pipeline_credito import clean_data, add_feature, verify_originals

if __name__ == "__main__":
    add_feature(clean_data())
    verify_originals()
    print("Feature criada na cópia derivada.")
