## Energy
- No necesita entrenarse.
- Score de la forma:$$E(x;f)=−T⋅log(\sum_{k=1}^{C}​e^{f_k​(x)/T})$$
- Si es ID, la suma de exponenciales debería ser alta, entonces el score debe ser bajo. Si es OOD, el modelo duda, entonces suma de exponenciales es chica, dando score negativo cercano a 0.
### Parms
- Temperature (t)

## MaxLogit
- No necesita parámetros extra.
- No necesita entrenarse
- El score lo da el inverso del logit máximo:$$ score(x)=-max_C \{f_{C}(x)\}$$
## Mahalanobis
- Necesita entrenamiento para fitear la GMM. Para cada clase del train se calcula la media y una matriz de cov compartida. Para cada nuevo ejemplo se calcula dist. de Mahalanobis entre vector de características y centro de la clase. 
- Score es el valor más bajo de distancias $$−max_{k}​\{(f(x)−μ_k​)^{⊤}Σ^{−1}(f(x)−μ_k​)\}$$
- Si un nuevo output tiene features muy distintas de las medias de clase, debería ser calificado como OOD
### Params
- eps: Se usa en inferencia y es simplemente una pequeña variación que debería empujar a scores ID a subir y scores OOD bajar (mayor separabilidad). (0 si se quiere Mahalanobis puro, en otro caso pasa a ser ODIN)
- norm_std [List]: Es un vector de stds para desnormalizar gradientes en ODIN. (stds usadas para norm imgs).
## MSP
- No necesita entrenamiento.
- Score calculado de la forma: $$score=−max_y​\{ σ_y​(\frac{f(x)}{T}​)\}$$
- Muestras ID deberían producir una prob. alta; mientras más baja sea la max probabilidad, más probable es que sea OOD la muestra.
### Params
- Temperature (t)
## KL-Matching
- No necesita parms. extra.
- Necesita entrenarse: calculo de distribución promedio para cada clase (usando data de **validación**). Obtiene logits para c/muestra, aplica softmax, clasifica y acumula distribuciones bajo cada clase. Finalmente promedia para obtener la dist.
- La idea es solucionar escenarios con muchas clases poco abarcables por MSP, ya que diferentes clases pueden mostrar distintas formas de distribución posterior. Luego una prob. baja no siempre indicaría OOD, si la clase tiene dist. más dispersa.
- Score de la forma: $$score(x)=D_{KL}​[p(⋅∣x)∥dy​]$$; donde el valor de la divergencia KL indica cuan diferente es la dist. posterior de la muestra respecto a la clase.


## React
- No necesita entrenamiento.
- Se plantea como una transformación de activaciones que se puede aplicar a un modelo antes de usar otro detector. La idea es que modelos pueden dar predicciones muy confiadas incluso para data OOD porque activaciones internas pueden alcanzar valores altos en OOD. Entonces se observó que muchas activaciones internas tienden a tener valores positivos muy grandes en datos OOD. la idea sería clipearlas para reducir confianza de modelo en OODs y así mejorar la separación.
### Params
- backbone: Primera parte del modelo, debería dar feature maps.
- head: Segunda parte del modelo, debería dar como output los logits a partir de mapa de características.
- threshold: corte de activaciones. Define límite máximo absoluto para cada elemento de la activación (clipping).
- detector: mapea outputs a algun OOD detector (ej. EnergyBased.score) 
## ViM
- Necesita entrenamiento para computar s.e. principal y alpha. Se calcula s.ee principal calculando matriz de covariancias de features centradas y se obtienen eigenvectors y eigenvalues. La idea es enconrtar direcciones principales en que features varían (d mayores eigenvalues). Esto representaría el espacio de mayor variación de los datos ID. El alpha es un escalador tal que el logit virtual tenga escala compatible con logits simplemente.
- La idea es combinar espacio de caracterśiticas de modelo y logits. 
- score: Se proyecta mapa de caracteristicas en s.e. ID y se calcula el residuo:$$\tilde{f}= f(x)-u​$$; con u un vector centrado que elimina bias de clasificador. Luego residuo (r) queda:$$r=\tilde{f}​−f_{proj}$$ y se toma norma $||r||_2$. Un residuo pequeño indicaría ID, mientras que uno grande sería OOD. Con esto se define el logit virtual:$$z_{OOD}=\alpha \cdot ||r||_2$$ Luego se toma el max logit real y se calcula el score ViM como sigue:$$score_{ViM}=log(\sum_{k=1}^{C}e^{f_k(x)})-z_{OOD}$$ Si energía es baja (incertidumbre de clases) y/o residuo es alto (lejos de s.e principal),el score es bajo.


### Params
- d: Dimensionalidad de subespacio principal (no residual).
- w: pesos de última layer.
- b: biases de última layer.
