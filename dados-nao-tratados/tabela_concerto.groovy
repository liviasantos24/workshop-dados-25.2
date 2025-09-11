import panda as pd

df = pd.read_csv("/dados_nao_tratados.csv")
df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
df['data_inscricao'] = pd.to_datetime(df['data_inscricao'], errors='coerce')
df['nota'] = pd.to_numeric(df['nota'], errors='coerce')
df['nota'] = df['nota'].fillna(10) 
df['ativo'] = df['ativo'].str.strip().str.lower().map({
    'sim': True, 'yes': True, 'true': True,
    'não': False, 'nao': False, 'n': False, 'false': False, 'e': False
})
df['idade'] = df['idade'].fillna(19)
df['data_inscricao'] = df['data_inscricao'].fillna('nulo')
df
