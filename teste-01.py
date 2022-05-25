import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
# módulos necessário para a tela "diabetes":
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


st.sidebar.title('Menu')

op = st.sidebar.selectbox('Selecione a página',['Página 1', 'Página 2', 'Página 3', 'Página 4'])

if op=='Página 1':
# referências:
	# https://medium.com/data-hackers/desenvolvimento-de-um-aplicativo-web-utilizando-python-e-streamlit-b929888456a5
	# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
	#dados dos usuários com a função
	def get_user_date():
		pregnancies = st.sidebar.slider("Gravidez",0, 15, 1)
		glicose = st.sidebar.slider("Glicose", 0, 200, 110)
		blood_pressure = st.sidebar.slider("Pressão Sanguínea", 0, 122, 72)
		skin_thickness = st.sidebar.slider("Espessura da pele", 0, 99, 20)
		insulin = st.sidebar.slider("Insulina", 0, 900, 30)
		bni= st.sidebar.slider("Índice de massa corporal", 0.0, 70.0, 15.0)
		dpf = st.sidebar.slider("Histórico familiar de diabetes", 0.0, 3.0, 0.0)
		age = st.sidebar.slider ("Idade", 15, 100, 21)
		#dicionário para receber informações
		user_data = {'Gravidez': pregnancies,
		'Glicose': glicose,
		'Pressão Sanguínea': blood_pressure,
		'Espessura da pele': skin_thickness,
		'Insulina': insulin,
		'Índice de massa corporal': bni,
		'Histórico familiar de diabetes': dpf,
		'Idade': age
		}
		features = pd.DataFrame(user_data, index=[0])
		return features

	# df = pd.read_csv('/home/agastesi/lab/python/indoor/streamlit_test/diabetes.csv')
	df = pd.read_csv('diabetes.csv')
	st.title("Prevendo Diabetes")
	st.subheader("Informações dos dados")
	#nomedousuário
	user_input = st.sidebar.text_input("Digite seu nome")
	st.write("Paciente:", user_input)
	#dados de entrada
	x = df.drop(['Outcome'],1)
	y = df['Outcome']
	#separa dados em treinamento e teste
	x_train, x_text, y_train, y_test = train_test_split(x, y, test_size=0.2)
	# Chamando a função criada e gerar um gráfico para exibir as informações
	user_input_variables = get_user_date()
	#grafico
	graf = st.bar_chart(user_input_variables)
	dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
	# dtc = decision_tree_classifier(criterion='entropy', max_depth=3)
	dtc.fit(x_train, y_train)
	#acurácia do modelo
	st.subheader('Acurácia do modelo')
	st.write(accuracy_score(y_test, dtc.predict(x_text))*100)
	#previsão do resultado
	prediction = dtc.predict(user_input_variables)
	st.subheader('Previsão:')
	st.write(prediction)


elif op=='Página 2':
	df = pd.read_csv("/home/agastesi/lab/python/indoor/circ_new/db/_FILES.csv")
	st.subheader("Todas as midias do Criatv:")
	st.write(df)
elif op=='Página 3':
	st.title('Página 1 selecionada')
	st.favicon=":shark:"
	st.selectbox('Selecione uma Opção',['Opção 1','Opção 2'])
	# ------------------------
	with st.form("key=my_form"):
		st.write("Inside the form")
		username = st.text_input("username")
		password = st.text_input("password")
		slider_val = st.slider("Form slider")
		checkbox_val = st.checkbox("Form checkbox")
		# Every form must have a submit button.
		submitted = st.form_submit_button("Submit")
		if submitted:
			st.write("slider", slider_val, "checkbox", checkbox_val)
			st.write(username, password)
	st.write("Outside the form")
	# ------------------------
	with st.echo():
		st.write('This code will be printed')

elif op=='Página 4':
	st.title('Página 2 selecionada')
	st.write('Hello, *World!* :sunglasses:')
	df = ({
		'first column': [1, 2, 3, 4, 5],
		'second column': [10, 20, 30, 40, 50],
	})

	df = pd.DataFrame(df)
	st.write(df)

	# ---------------------
	df = pd.DataFrame(
	np.random.randn(200, 3),
	columns=['a', 'b', 'c'])
	c = alt.Chart(df).mark_circle().encode(
	x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
	st.write(c)
	# ------------------------
	df = pd.DataFrame(
   	np.random.randn(50, 20),
	columns=('col %d' % i for i in range(20)))
	st.dataframe(df)  # Same as st.write(df)
	# --------------------
	bt01 = st.button('Click me')
	bt02 = st.download_button('Click me','file')
	# --------------------
	# for i in range(100):
	# 	st.progress(i)
	# 	time.sleep(.25)
	# --------------------
	st.balloons()