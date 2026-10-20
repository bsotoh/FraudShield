# Ficha de la base de datos

**Certamen 1 — Machine Learning**

**Nombre del proyecto:** FraudShield — Detección de fraudes en transacciones financieras

### 1. Identificación de la base de datos

| Característica | Descripción |
|---|---|
| **Nombre del dataset** | IEEE-CIS Fraud Detection |
| **Fuente** | Kaggle — IEEE Computational Intelligence Society |
| **Enlace** | https://www.kaggle.com/competitions/ieee-fraud-detection/data |
| **Archivo utilizado** | `train_transaction.csv` |
| **Número de registros** | 590.540 transacciones |
| **Número de columnas originales** | 394 |
| **Variables predictoras utilizadas** | 392 |
| **Variable objetivo** | `isFraud` |
| **Tipo de problema** | Clasificación binaria desbalanceada |

### 2. Distribución de clases

| Clase | Significado | Registros | Proporción |
|---|---|---:|---:|
| 0 | Transacción legítima | 569.877 | 96,50 % |
| 1 | Transacción fraudulenta | 20.663 | 3,50 % |
| **Total** | | **590.540** | **100 %** |

La clase minoritaria corresponde a las transacciones fraudulentas, con aproximadamente un 3,5 % del total. Por lo tanto, la base cumple el requisito de desbalance establecido para el certamen.

### 3. Contexto y propósito del proyecto

Nuestro proyecto FraudShield busca desarrollar un sistema de apoyo para detectar posibles fraudes en transacciones financieras. El objetivo es identificar operaciones sospechosas y reducir los costos económicos asociados a los errores de clasificación.

En este contexto, un **falso positivo** corresponde a una transacción legítima que genera una alerta innecesaria, mientras que un **falso negativo** representa un fraude que no logramos detectar.

Para evaluar estas consecuencias, definimos costos hipotéticos de $5.000 CLP por falso positivo y $100.000 CLP por falso negativo.

### 4. Metodología propuesta

Compararemos cinco algoritmos de clasificación: Árbol de Decisión, Random Forest, XGBoost, AdaBoost y LightGBM, utilizando SMOTENC y pesos de clase para abordar el desbalance.

Seleccionaremos el modelo mediante validación cruzada y analizaremos sus errores desde una perspectiva económica para determinar un umbral de clasificación conveniente.

Los resultados se presentarán mediante un dashboard interactivo que permitirá modificar los costos, comparar estrategias y visualizar el ahorro económico estimado.

### 5. Recursos del proyecto

**Repositorio GitHub:**  
https://github.com/bsotoh/FraudShield

**Dashboard interactivo:**  
https://fraudshield-machinelearning.streamlit.app
