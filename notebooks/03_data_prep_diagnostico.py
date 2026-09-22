"""Limpeza estrutural vigente; os originais são somente leitura."""
from pipeline_credito import clean_data, verify_originals

if __name__ == "__main__":
    clean_data()
    verify_originals()
    print("Limpeza concluída; imputação estatística fica para o treino.")
