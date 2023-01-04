import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


global files
files = st.sidebar.file_uploader("Cargar datos",type=['csv','xls', 'xlsx','json'],accept_multiple_files=False)

def load_table():
    if files is not None:
        if files.type  == 'text/csv':
            df = pd.read_csv(files,sep=',|;',engine="python")
            return df
        elif files.type == 'application/vnd.ms-excel' or files.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' :
            df = pd.read_excel(files)
            return df
        elif files.type == 'application/json':
            df = pd.read_json(files)
            return df
        else:
            st.error('Formato de archivo no soportado')
            return None

def main():
    continuar = True
    st.title('Proyecto2')
    if files is None:
        st.warning('No se ha cargado ningún archivo')
    else:
        st.info('Archivo cargado:  '+ files.name)
        table = load_table()
        with st.expander("Datos ingresados", expanded=True):
            st.write('Datos cargados: ')
            st.dataframe(table,width=800,height=400)
        header = []
        ## verficiar si el archivo tiene encabezado
        if table.columns.to_list() is None:
            continuar = False
            st.warning('El archivo no tiene encabezado, por favor ingrese los encabezados')
        else:
            ## obtener cabecera del archivo
            header  = table.columns.to_list()
        if continuar:
            #regresion lineal 
            with st.expander("Regresion Lineal", expanded=False):
                try:
                    st.write("Regresion Lineal")
                    ## variable dependiente
                    var_y = st.selectbox('Seleccione una variable dependiente (y)',header)
                    var_x = st.selectbox('Seleccione una variable independiente (x)',header)
                    ## verificar si la variable dependiente es la misma que la independiente
                    if var_x == var_y:
                        st.error('La variable dependiente no puede ser la misma que la independiente')
                    else:
                        # obtener columnas 
                        x = np.asanyarray((table[var_x]).values.reshape(-1, 1))
                        y = table[var_y]
                        # predicciony , coeficiente , intercepcion , r^2 , errror cuadratico
                        regr = LinearRegression()
                        regr.fit(x, y)
                        y_pred = regr.predict(x)
                        r2 = r2_score(y, y_pred)
                        # datos actuales
                        st.markdown("**Datos**")
                        actuales = pd.DataFrame(x,y_pred)
                        st.dataframe(actuales,width=300,height=400)
                        st.markdown("**Resultados**")
                        st.markdown("  Coeficiente:  ```"+str(regr.coef_)+"    ``` ")
                        st.markdown("  Intercepcion:  ```"+str(regr.intercept_ )+"  ``` ")
                        st.markdown(" Coeficiente de Determinación R2: ``` "+str(r2)+"   ``` ")
                        st.markdown(" Error Cuadrático Medio (MSE): ```  "+str(mean_squared_error(y, y_pred))+"   ``` ")
                        inter = "+"+str(regr.intercept_ ) if regr.intercept_ > 0 else str(regr.intercept_ )
                        st.markdown("**Función**")
                        st.latex ('f(x)= '+str(regr.coef_[0])+'x '+ inter+'   ')
                        # prediccion
                        st.markdown("**Prediccion**")
                        prediccion = st.number_input('Ingrese un valor de prediccion',value=0)
                        st.markdown(" Prediccion : ``` "+str( regr.predict([[prediccion]]) ) +"  ``` ") 
                        # grafica de la regresion lineal
                        st.markdown("**Grafica de la regresion lineal**")
                        figlineal, ax = plt.subplots(figsize=(10, 4))
                        ax.scatter(x, y, color='blue')
                        ax.plot(x, y_pred, color='red', linewidth=1)
                        plt.xlabel(var_x)
                        plt.ylabel(var_y)
                        plt.title('Regresion Lineal')
                        st.pyplot(figlineal) 
                    
                    
                except Exception as e:
                    print(e)
                    st.error('Error al ejecutar el algoritmo')
            
            # regresion polinomial
            with st.expander("Regresion polinomial", expanded=False):
                st.write("Regresion Polinomial")
                varP_y = st.selectbox('  Seleccione una variable dependiente (y) ',header)
                varP_x = st.selectbox('  Seleccione una variable independiente (x) ',header)
                ## verificar si la variable dependiente es la misma que la independiente
                if varP_x == varP_y:
                    st.error('La variable dependiente no puede ser la misma que la independiente')
                else:
                    # preguntar grado de polinomio
                    grado = st.number_input('Grado de polinomio', value=2)
                    pf = PolynomialFeatures(degree=grado)
                    try: 
                        x = np.asanyarray((table[varP_x]).values.reshape(-1, 1))
                        y = table[varP_y]
                        x_trans = pf.fit_transform(x)
                        # predicciony , coeficiente , intercepcion , r^2 , errror cuadratico
                        regr = LinearRegression()
                        regr.fit(x_trans, y)
                        #recorrer coeficientes
                        resultado = ""
                        gaux = grado
                        for c in regr.coef_:
                            if gaux > 1:
                                cc =  "+"+str(c) if c > 0 else str(c)
                                resultado = resultado + cc + "x^" + str(gaux) 
                                gaux = gaux - 1
                            elif gaux == 1:
                                cc =  "+"+str(c) if c > 0 else str(c)
                                resultado = resultado + cc + "x^" + str(gaux)
                                gaux = gaux - 1
                            else:
                                cc =  "+"+str(c) if c > 0 else str(c)
                                resultado = resultado + cc
                        if resultado[0] == "+":
                            resultado = resultado[1:]
                        y_pred = regr.predict(x_trans)
                        rmse = np.sqrt(mean_squared_error(y, y_pred))
                        r2 = r2_score(y, y_pred)
                        #st.write ("coeficientes: " +regr.coef_)
                        #print("coeficientes: " +regr.coef_)
                        # resultados
                        st.markdown("**Resultados**")
                        st.markdown(" Coeficiente de Determinación R2:  ```"+str(r2)+"   ``` ")
                        st.markdown(" Error Cuadrático Medio (MSE): ```  " +str(mean_squared_error(y, y_pred))+"   ``` ")
                        st.markdown(" Raíz del Error Cuadrático Medio (RMSE): ```  " +str(rmse)+"   ``` ")
                        st.markdown("**Función**")
                        st.latex ('f(x)= '+resultado)
                        # prediccion
                        st.markdown("**Prediccion**")
                        pPolinomial = st.number_input('Ingrese un valor de prediccion ', value=0)
                        x_new_min = pPolinomial 
                        x_new_max = pPolinomial 
                        x_new = np.linspace(x_new_min, x_new_max, 1)
                        x_new = x_new[:, np.newaxis]
                        x_trans = pf.fit_transform(x_new)
                        st.markdown(" Prediccion : ```" +str(regr.predict(x_trans)) + "  ``` ")
                        # grafico
                        st.markdown("**Gráfica**")
                        fig = plt.figure(figsize=(10, 4))
                        plt.scatter(x, y, color='blue')
                        plt.plot(x, y_pred, color='red', linewidth=3)
                        st.pyplot(fig)
                    except Exception as e:
                        print(e)
                        st.error('Esperando ...')
            # clasificador gausiano 
            with st.expander("Clasificador Gausiano", expanded=False):
                st.write("Clasificador Gausiano")
                #obtener las columnas a evaluar
                bandera = True
                tabla2 = table.copy()
                aux_y = st.selectbox('Seleccione la clase:',header)
                aux_x = st.multiselect('Seleccione el conjunto de datos de entrenamiento:',header, header)
                      
                if aux_y in aux_x:
                    st.error('La clase no puede estar entre el conjunto de datos de entrenamiento')
                    bandera = False
                if bandera :
                    try:
                        codEtiquetas = []
                        arryAxu = []
                        le = preprocessing.LabelEncoder()
                        for item in aux_x:
                            aux =tabla2[item]
                            arryAxu.append(le.fit_transform( aux))
                            codEtiquetas.append(dict(zip(aux, le.fit_transform( aux))))

                        # agregar clase 
                        labelGauss = np.array (le.fit_transform(tabla2[aux_y]))
                        codEtiquetas .append(dict(zip(tabla2[aux_y], labelGauss)))
                        print("agregando codificacion de  etiquetas ")
                        print(codEtiquetas)
                        liss = np.array(arryAxu)
                        features = list(zip(*liss))
                        #crear clasificador gaussiano
                        clf = GaussianNB()
                        clf.fit(features, labelGauss)
                        # crear dataframe con las etiquetas codificadas
                        st.markdown('**Etiquetas codificadas**')
                        diccionarioFinal ={}
                        for d in codEtiquetas:
                            diccionarioFinal.update(d)

                        st.dataframe([[key,int(diccionarioFinal[key])] for key in diccionarioFinal.keys()], width=500,height=300)
                        # recibir valores 
                        val_predicciones = st.text_input('Ingrese los valores numericos de las etiquetas codificadas, separado por comas.')
                        # verificar el tamaño
                        val =  val_predicciones.split(",")
                        if len(val)  == len(aux_x):
                            #verificar que sean de tipo string 
                            print (val)
                            try:
                                int_val = list(map(int, val))
                            except:
                                st.error('Los valores deben ser de tipo entero o flotante')
                                bandera = False
                            if bandera:
                                # resultados
                                st.markdown("**Resultados**")
                                st.markdown(" ```Resultado de Prediccion: "+str(clf.predict([int_val]))+"   ``` ")
                        else:
                            st.error('El numero de valores no corresponde al numero de columnas seleccionadas')
                    except Exception as e:
                        print(e)
                        st.error('Error al  convertir valores de variables independientes, seleccione las columas correctas')
            
            with st.expander("Clasificador de árboles de decisión", expanded=False):
                st.write("Clasificador de árboles de decisión")
                #obtener las columnas a evaluar
                banderaID3 = True
                tablaID3 = table.copy()
                #obtener la clase 
                id3_y = st.selectbox('Seleccione la clase:  ',header)
                id3_x = st.multiselect('Seleccione el conjunto de datos de entrenamiento',header, header)           
                if id3_y in id3_x:
                    st.error('La clase no puede estar entre el conjunto de datos de entrenamiento')
                    banderaID3 = False
                if banderaID3 :
                    try:
                        arryAxu = []
                        le = preprocessing.LabelEncoder()
                        for item in id3_x:
                            aux = tablaID3[item]
                            arryAxu.append(le.fit_transform( aux))
                        # agregar clase 
                        label = np.array (le.fit_transform(tablaID3[id3_y]))
                        lis = np.array(arryAxu)
                        print(lis[1])
                        feat = list(zip(*lis))
                        # fit model
                        st.markdown("**Gráfica**")
                        clf =  DecisionTreeClassifier().fit(feat,label)
                        fig2 = plt.figure(figsize=(10, 5))
                        plot_tree(clf,filled= True)
                        st.pyplot(fig2)                                   
                    except Exception as e:
                        st.error('Error al ejecutar el algoritmo, revise  los datos del archivo de entrada ')
                        print (e)

            #redes neuronales
            with st.expander("Redes neuronales", expanded=False):
                try:
                    st.write("Redes Neuoranles")
                    ## variable dependiente
                    varR_y = st.selectbox('Seleccione una variable (y)',header)
                    varR_x = st.selectbox('Seleccione una variable (x)',header)
                    ## verificar si la variable dependiente es la misma que la independiente
                    if varR_x == varR_y:
                        st.error('La variable dependiente no puede ser la misma que la independiente')
                    else:
                        st.write("Redes Neuoranles")
                    ## variable dependiente
                    varR_y = st.selectbox('Seleccione una variable (y)',header)
                    varR_x = st.selectbox('Seleccione una variable (x)',header)
                    ## verificar si la variable dependiente es la misma que la independiente
                    if varR_x == varR_y:
                        st.error('La variable dependiente no puede ser la misma que la independiente')
                    else:
                        # obtener columnas 
                        #x = np.asanyarray((table[varR_x]).values.reshape(-1, 1))
                        x = table[varR_x]
                        y = table[varR_y]
                        # print("la coumna x: ", x)
                        # print("la coumna y: ", y)
                        
                        X=x[:, np.newaxis]
                        X_train, X_test, y_train, y_test = train_test_split(X, y)
                       
                        mlr = MLPRegressor(solver='lbfgs', alpha= 1e-5, hidden_layer_sizes=(3,3), random_state=1)
                        mlr.fit(X_train, y_train)
                        #print(mlr.score(X_train, y_train))
                        #print("PREDICCION", mlr.predict(20))
                        st.markdown("IMPRESION  "+str(mlr.score(X_train, y_train))+"     ")

                  
                    
                except Exception as e:
                    print(e)
                    st.error('Error al ejecutar el algoritmo')



if __name__ == '__main__':
    main()