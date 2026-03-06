<p align="center">
  <img src="assets/banner.png" alt="DM-SUS-Analytics Banner" width="100%"/>
</p>

<h1 align="center">DM-SUS-Analytics</h1>
<p align="center"><b>Plataforma de Análise de Saúde Pública Municipal — SUS</b></p>
<p align="center">
  <img src="https://img.shields.io/badge/Autor-Eduardo%20Muniz%20Alves-blue"/>
  <img src="https://img.shields.io/badge/Empresa-DM%20Technology-green"/>
  <img src="https://img.shields.io/badge/Licença-Apache%202.0-red"/>
  <img src="https://img.shields.io/badge/Python-3.9+-yellow"/>
  <img src="https://img.shields.io/badge/SUS-Saúde%20Pública-brightgreen"/>
</p>

---

## Português 🇧🇷

### Sobre

Plataforma de análise e otimização da saúde pública municipal integrada ao SUS (Sistema Único de Saúde). Calcula indicadores epidemiológicos conforme metodologia RIPSA/Ministério da Saúde, analisa cobertura da Atenção Básica, e otimiza alocação de recursos com programação linear.

### Indicadores Implementados

#### Mortalidade

| Indicador | Fórmula | Referência Brasil | Meta ODS 2030 |
|-----------|---------|-------------------|---------------|
| **TMG** (Taxa Mortalidade Geral) | `(Óbitos / Pop) × 1.000` | 6,5/1.000 hab | — |
| **TMI** (Taxa Mortalidade Infantil) | `(Óbitos <1a / NV) × 1.000` | 11,9/1.000 NV | ≤ 12 |
| **RMM** (Razão Mortalidade Materna) | `(Óbitos maternos / NV) × 100.000` | 55/100.000 NV | < 70 |
| **APVP** (Anos Potenciais Vida Perdidos) | `Σ (70 - idade_óbito)` | — | — |

#### Atenção Básica

| Indicador | Fórmula | Meta |
|-----------|---------|------|
| **Cobertura ESF** | `(Equipes × 3.450 / Pop) × 100` | 100% |
| **Cobertura Vacinal** | `(Doses / Pop-alvo) × 100` | ≥ 95% |
| **Taxa ICSAB** | `(Internações CSAB / Pop) × 10.000` | < 100/10.000 |

### Classificação de Indicadores

```
[BOM]      → Dentro ou abaixo da meta
[REGULAR]  → Próximo da meta, requer atenção
[RUIM]     → Acima da meta, intervenção necessária
[CRÍTICO]  → Emergência, ação imediata
```

### Referências Normativas

- **RIPSA** — Rede Interagencial de Informações para a Saúde
- **Portaria MS/GM nº 3.947/1998** — Indicadores básicos de saúde
- **Portaria SAS/MS nº 221/2008** — Lista de ICSAB
- **Portaria nº 2.436/2017** — Política Nacional de Atenção Básica
- **ODS 3** — Saúde e Bem-Estar (Agenda 2030 ONU)

### Instalação

```bash
pip install dm-sus-analytics
```

### Uso Rápido

```python
from dm_sus.indicadores.epidemiologicos import *

# Dados do município
pop = PopulacaoMunicipal(
    codigo_ibge="3550308", nome="São Paulo", uf="SP",
    populacao_total=12_300_000, nascidos_vivos=160_000
)

obitos = DadosObito(
    total_obitos=78_000,
    obitos_menores_1=1_600,
    obitos_maternos=88
)

# Calcular indicadores
calc = CalculadoraMortalidade()
tmi = calc.taxa_mortalidade_infantil(obitos, pop)
print(f"TMI: {tmi.valor} {tmi.unidade} [{tmi.classificacao}]")

# Otimizar alocação de equipes ESF
opt = OtimizadorRecursos()
resultado = opt.alocar_equipes_esf(pop.populacao_total, orcamento=200_000_000)
print(f"Equipes: {resultado['equipes_alocadas']}, Cobertura: {resultado['cobertura_estimada']}%")
```

---

## English 🇺🇸

### About

Municipal public health analytics platform integrated with Brazil's SUS (Unified Health System). Computes epidemiological indicators following RIPSA/Ministry of Health methodology, analyzes Primary Care coverage, and optimizes resource allocation using linear programming.

### Key Features

- **Mortality Indicators**: General mortality rate, infant mortality rate, maternal mortality ratio, YPLL
- **Primary Care**: Family Health Strategy (ESF) coverage, vaccination coverage, ACSC hospitalization rates
- **Resource Optimization**: Optimal ESF team allocation given budget constraints
- **Investment Prioritization**: Automatic ranking of intervention areas by indicator severity

### Quick Start

```python
from dm_sus.indicadores.epidemiologicos import *

pop = PopulacaoMunicipal("3550308", "São Paulo", "SP", 12_300_000, nascidos_vivos=160_000)
obitos = DadosObito(78_000, obitos_menores_1=1_600, obitos_maternos=88)

calc = CalculadoraMortalidade()
tmi = calc.taxa_mortalidade_infantil(obitos, pop)
# TMI: 10.0 per 1,000 live births [bom]
```

---

**Autor:** Eduardo Muniz Alves | **Empresa:** DM Technology | **Licença:** Apache 2.0
