# Política de integración de la auditoría de cierre

La referencia candidata sigue siendo única: `data/qrels_candidates_for_approval.tsv`. Todas sus filas permanecen pendientes de aprobación humana y `data/qrels.tsv` no se sustituye automáticamente.

## Uso de fuentes completas

Se utiliza el artículo completo únicamente cuando la auditoría aporta una URL, el título y el asunto coinciden con el documento del corpus, y la página permite atribuir evidencia inequívoca al mismo artículo. La decisión registra `evidence_scope=articulo_completo_externo`, la URL y una justificación que declara expresamente que supera el extracto almacenado. Las tres fuentes se consultaron el 23 de octubre de 2026.

Para resolver un caso como relevancia 2, el artículo completo debe situar en el centro de la noticia el criterio de grado 2. En vigilancia debe identificar además una autoridad pública o vigilancia respaldada por el Estado y una práctica concreta de recopilación, análisis o rastreo. En cricket debe desarrollar una controversia institucional, de gobernanza, disciplina, política o seguridad.

Los demás juicios se basan exclusivamente en el título y texto original conservado en el corpus. La ausencia de evidencia en el extracto no se reemplaza con inferencias externas no registradas.

## Requisito de 10–15 documentos

`data/academic_reference_core.tsv` es sólo un índice expositivo de ejemplos ya contenidos en la referencia ampliada. No es una segunda referencia, no reemplaza ningún qrels y no debe emplearse como denominador de Recall@10. Las métricas futuras deberán usar todos los positivos de la referencia única una vez aprobada.
