# Tratamiento del requisito académico de 10–15 relevantes

Existe una sola referencia candidata para evaluación: `qrels_candidates_for_approval.tsv`, que conserva todos los positivos auténticos y sigue pendiente de aprobación humana.

`academic_reference_core.tsv` identifica hasta 15 ejemplos previamente adjudicados por necesidad (aplicables a sus tres consultas) únicamente como núcleo documental para exposición y auditoría. No es un segundo qrels, no está completo y no debe alimentar métricas.

| Necesidad | Positivos completos | Núcleo documental | Recall@10 máximo con referencia completa |
|---|---:|---:|---:|
| trade | 28 | 15 | 35.7% |
| energy | 17 | 15 | 58.8% |
| football | 19 | 15 | 52.6% |
| cricket | 13 | 13 | 76.9% |
| surveillance | 25 | 15 | 40.0% |

Usar sólo el núcleo como denominador inflaría artificialmente Recall@10 y rompería la comparabilidad. Por ello las métricas futuras deben calcularse exclusivamente sobre la referencia completa una vez aprobada; el núcleo sólo documenta el cumplimiento expositivo del intervalo solicitado.
