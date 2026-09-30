## Pipeline TESA (Rogasch et al., 2017)

Esta é a sequência recomendada no manual e no artigo do TESA, com a ICA feita em duas rodadas:

| # | Etapa | Função TESA |
|---|---|---|
| 1 | Localizar os pulsos de TMS | `tesa_findpulse` |
| 2 | Remover a janela do pulso (ex.: −2 a 10 ms) | `tesa_removedata` |
| 3 | Interpolar a janela removida (cúbica), necessário antes de fazer downsample | `tesa_interpdata` |
| 4 | Downsample (ex.: 5 kHz → 1 kHz) | `pop_resample` |
| 5 | Epocar (−1 a 1 s) e corrigir a linha de base | `pop_epoch`, `pop_rmbase` |
| 6 | Remover canais e trials ruins | manual ou `pop_rejkurt` |
| 7 | **Zerar a janela interpolada** antes da ICA | `tesa_removedata` |
| 8 | **ICA 1**: remover o artefato muscular grande evocado pela TMS | `tesa_fastica`, `tesa_compselect` |
| 9 | Interpolar a janela removida de novo (cúbica) | `tesa_interpdata` |
| 10 | **Filtrar**: passa-banda 1–100 Hz e rejeita-banda 48–52 Hz | `tesa_filtbutter` |
| 11 | Zerar a janela interpolada de novo | `tesa_removedata` |
| 12 | **ICA 2**: piscadas, movimentos oculares, músculo persistente, decaimento | `tesa_fastica`, `tesa_compselect` |
| 13 | Interpolar a janela e os canais removidos | `tesa_interpdata`, `pop_interp` |
| 14 | Rereferenciar para a média e corrigir a linha de base | `pop_reref` |

## Pontos relevantes para a sua refatoração

1. **O filtro do TESA não é FIR.** O `tesa_filtbutter` usa um **Butterworth IIR de ordem 4, de fase zero** (`filtfilt`) e aplica um rejeita-banda de 4 Hz. Por isso a largura de 4 Hz do seu `band=(58, 62)` bate com o TESA adaptado para a rede de 60 Hz.
2. **A filtragem só acontece depois da ICA 1.** O artefato muscular grande sai antes de filtrar, e é isso que reduz o ringing. A interpolação cúbica sozinha não é suficiente, e o TESA também parte desse princípio.
3. **Os harmônicos não são filtrados.** O passa-baixa de 100 Hz já remove 120 Hz em diante, e a rede fica coberta apenas pelo notch de 60 Hz.
4. **Existem alternativas no lugar da ICA 1:** SOUND e SSP-SIR (`tesa_sound`, `tesa_sspsir`), que também são aplicadas antes da filtragem.

## Equivalente no MNE

```python
raw.notch_filter(60, method="iir",
                 iir_params=dict(order=4, ftype="butter"),
                 notch_widths=4, phase="zero")
```

Na refatoração, sugiro oferecer `method="iir"` como opção compatível com o TESA e manter o notch **depois** da remoção de artefatos por ICA, SOUND ou SSP-SIR.