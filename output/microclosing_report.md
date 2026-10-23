# Microcorrección final de la referencia candidata

La integración añade dos pares al pool existente sin reconstruirlo y conserva una única referencia candidata en `data/qrels_candidates_for_approval.tsv`.

## Decisiones

- `surveillance / Nbc0999a090a07dc280d0`: relevancia 2. El texto original del corpus documenta desvío de tráfico de Internet de AT&T hacia la NSA y una acción judicial contra la interceptación gubernamental.
- `football / N5caa0ad89d6f0b141f37`: relevancia 2. La decisión usa el artículo completo de Global Voices, consultado el 23 de octubre de 2026, porque el extracto local no contiene la controversia. El artículo documenta actuación de autoridades libias, conflicto entre federaciones, retirada por seguridad, investigación y sanciones de CAF.
- `surveillance / N318aa6952b490b0a4b01`: relevancia 1 a 0. La observación aérea de asentamientos por FUNAI persigue evitar contacto y proteger territorio; el texto no desarrolla monitoreo de personas, tratamiento de datos personales ni consecuencias para la privacidad derivadas de esa actividad.

## Referencia y requisito académico

Los positivos completos por necesidad son: comercio 28, energía 14, fútbol 20, cricket 14 y vigilancia 25. Comercio, fútbol y vigilancia superan literalmente el intervalo académico de 10–15. No se eliminan positivos para cumplir una cuota.

`data/academic_reference_core.tsv` sigue siendo sólo un núcleo expositivo de hasta 15 ejemplos por necesidad. No constituye otro qrels, no reemplaza la referencia completa y no debe usarse para calcular Recall@10.

Todas las filas de `data/qrels_candidates_for_approval.tsv` conservan el estado `propuesta_ia_pendiente_aprobacion_humana`. `data/qrels.tsv` continúa provisional y no fue sustituido.
