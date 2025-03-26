#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install whoosh')


# In[1]:


from whoosh import index
import os
from whoosh.fields import Schema 
from whoosh.fields import TEXT # También se puede guardar tipos como NUMERIC, DATETIME, BOOLEAN

# si se almacena (stored=True) luego se pueden visualizar tal y como era el text original, sino
#solo se guardan su términos sueltos (pueden faltar términos como stopwords....)
esquema = Schema(autor=TEXT(stored=True), 
                titulo=TEXT(stored=True),
                cuerpo=TEXT(stored=True),
                )


# In[2]:


#crear un carpeta para guardar el índice
if not os.path.exists("directorio_indice"):
    os.mkdir("directorio_indice")
    
#crear el índice (si existe, lo sobrescribe)
index.create_in("directorio_indice", esquema) 
#abrir el índice
indice = index.open_dir("directorio_indice")
#crear un objeto escritor para escribir
writer = indice.writer()

#Añadir uno o más documentos índice
writer.add_document(
                autor=u"Pepito Perez",
                titulo=u"El contenido de la materia gris", #preprocesar la cadena si se considera necesario
                cuerpo= u"Aquí iría el contenido principal" )

#cerrar el escritor y el índice guardando todo lo escrito
writer.commit()


# In[3]:


from whoosh.qparser import *

#Definir un parser para un campo concreto
parserCuerpo=QueryParser("cuerpo", schema=esquema) 

#Parsear la cadena para convertirla a un objeto consulta (query)
consulta=parserCuerpo.parse(u"contenido OR especializado  OR hoteles") #otros operadores AND NOT ()
print (consulta)


# In[4]:


#búsqueda en múltiples campos
parserMultipleCampos = MultifieldParser(["titulo", "cuerpo"], schema=esquema)

#Parsear la cadena para convertirla a un objeto consulta (query)
consulta2=parserMultipleCampos.parse(u"contenido OR especializado  OR hoteles")
print (consulta2)


# In[5]:


from whoosh import scoring

# Abre el objeto buscador y luego lo cierra con el bloque "with"
with indice.searcher(weighting=scoring.TF_IDF()) as buscador:
    #Busca en el índice los documentos más parecidos devolviendo un máximo de documentos (limit)
    documentos_recuperados = buscador.search(consulta2, limit=20, terms = True) #terms = True guarda los términos que hicieron match entre la consulta y el documento
    
    #imprimir resultados
    for i in range(len(documentos_recuperados)):
        print(documentos_recuperados[i]['titulo'], 
              str(documentos_recuperados[i].score), documentos_recuperados[i]['cuerpo'])


# In[6]:


#Ver qué términos de la consulta han hecho match con los documentos
if documentos_recuperados.has_matched_terms():
    # Todos los términos que hicieron match
    print('Términos en toda la colección \n',documentos_recuperados.matched_terms())

    # Cada término en cada documento
    
    for i, doc in enumerate(documentos_recuperados, start = 1):
        print('Términos en doc #', i, doc.matched_terms())


# In[ ]:


#Otra forma de resaltar los términos que se han buscado y encontrado
from IPython.core.display import display, HTML
from whoosh import highlight

documentos_recuperados.fragmenter = highlight.SentenceFragmenter(charlimit=100000, maxchars=200)
#documentos_recuperados.fragmenter = highlight.WholeFragmenter()
for hit in documentos_recuperados:
    #print(hit['titulo'])
    display(HTML(hit.highlights("titulo", top=2)))
    display(HTML(hit.highlights("cuerpo", top=2)))

