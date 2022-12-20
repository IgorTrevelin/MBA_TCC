import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "files",
    type=str,
    nargs="+",
    help="Filename or filepath of the file to be processed",
)
args = parser.parse_args()

files = args.files

for file in files:

    with open(file, "r") as f:
        lines = f.readlines()

    translations = {
        "Open": "Abertura",
        "High": "Máximo",
        "Low": "Mínimo",
        "Close": "Fechamento",
        "Bitcoin USD Exchange Trade Volume": "Volume de Negociação do Bitcoin",
        "Bitcoin Cost Per Transaction": "Custo por Transação do Bitcoin",
        "Bitcoin Total Transaction Fees USD": "Total de Tx. de Transação do Bitcoin USD",
        "Bitcoin Hash Rate": "Tx. de Hashing do Bitcoin",
        "Bitcoin Number of Transactions": "Número de Transações do Bitcoin",
        "Direction": "Direção do Bitcoin",
        "Gold": "Ouro",
        "Silver": "Prata",
        "Copper": "Cobre",
        "Oat": "Aveia",
        "Sugar": "Açúcar",
        "Platinum": "Platina",
        "Natural Gas": "Gás Natural",
        "Palladium": "Paládio",
        "Crude Oil": "Óleo Cru",
        "Cocoa": "Cacau",
        "NASDAQ Future": "NASDAQ Futuro",
        "S&P500 Future": "S&P500 Futuro",
        "KOSPI Index": "KOSPI Índice Futuro",
    }

    items = [l.strip() for l in lines]
    items = [i.replace("_", " ") for i in items]
    for i in range(len(items)):
        for old, new in translations.items():
            items[i] = items[i].replace(old, new)

    items.sort(reverse=False)
    print(f"Output for file {file}")
    print(", ".join(items))
