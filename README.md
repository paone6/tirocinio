# Riconoscere la lingua parlata tramite movimenti articolatori delle labbra - Progetto Tirocinio Mario Paone

> L'obiettivo del progetto è quello di formare un dataset con video di soggetti che parlano frontalmente alla telecamera e utilizzarli per addestrare un classificatore che tramite il video, riconosca la lingua parlata dal soggetto tramite i soli 
movimenti articolatori delle labbra.

*There is no english version of this yet

# Indice

- [Come iniziare](#come-iniziare)
- [Come contribuire](#come-contribuire)
- [Manutenzione](#manutenzione)
- [Licenza](#licenza)

# Come iniziare
Vedere prima la sezione "come installare".
Una volta installato, il progetto funziona in questo modo:
1) Allo script 'VideoProducer.py' viene indicato il path di una cartella contenente i video originali, lo script si occupa di      salvare per ogni video orignale un sottoinsieme di video di 15 secondi l'uno in cui vi sia sempre il viso del soggetto ben visibile per un numero di secondi prestabilito.
Per modificare il numero di secondi contigui che ogni sequenza deve avere si deve modificare la costante CONTIGUOUS_SECONDS(che sia un divisore di 15)
Per modificare il numero massimo di video da 15 secondi prodotti dallo script si deve modificare la costante MAX_PER_VIDEO.
Per impostare la cartella sorgente dei video e quella di destinazione modificare path e destinationPath.
2)Una volta prodotti i video da 15 secondi si procede con lo script 'CSVProducerOnlyMouth' che si occupa di prendere i suddetti video e per ognuno di questi:
    -Stampa il video con solo le labbra senza i landmark
    -Stampa il video con solo le labbra ed i landmark 
    -Stampa il file csv contenente le distanze eucluidee tra i vari landmark 
Per decretare quali landmark prendere bisogna modificare l'istruzione alla riga 138 che può essere:
    - 'extern_mouth' prende solo i 12 landmark dell'esterno labbra
    - 'intern_mouth' prende gli 8 landmark dell'interno labbra
    - 'mouth' prende tutti e 20 i landmark delle labbra
Per modificare la cartella sorgente dei video e quelle di destinazione bisogna modificare i diversi path nel file.
3)Nel file 'my_dataset.py' va modificato il path della cartella contente i file csv che formeranno il nostro dataset. (Attenzione: è importante che il dataset sia bilanciato)
4)Tramite il file 'Classifier.ipynb' viene richiamato il metodo load_dataset() di my_dataset e viene testato il risultato con i diversi classificatori.

NB. I file presenti nella cartella che non sono descritto nei passaggi precedenti sono legati a tentativi di implementazione precedenti al quale è stata poi preferita la strategia descritta sopra. Per completezza in seguito vengono brevemente descritti.
CSVProducer: Stampa i file csv senza ritagliare solo le labbra, questo portava a problemi di normalizzazione, essendo la distanza dei soggetti nei video diversa tra loro. 
my_dataset_norm e my_dataset_norm2: Svolgono funzione simile a my_dataset, però una volta letti i file csv procedono con due tecniche differenti alla normalizzazione e standardizzazione dei dati.
PrintImageFromCSv: Dati i file csv, stampava un immagine con le distanze euclidee in essi presenti. Questo veniva fatto per seguire una strategia di classificatori che potesse riconsocere e classificare le immagini date. 

## Come installare
Una volta scaricato il programma è necessario modificare in tutti i file che lo necessitano, la locazione del file 'shape_predictor_68_face_landmarks.dat', necessario per riconsocere il viso e decretare i landmark che lo formano.
Inoltre è necessarios settare l'ambiente di sviluppo seguendo le istruzioni contenute a questo link:  https://drive.google.com/drive/u/0/folders/1giEuqg7W5vJc56fJP7u01dSROiC3l4Lb

## Documentazione
### Link a documentazione esterna 

# Come contribuire
Per contribuire al progetto eseguire una fork

## Autori e Copyright
    Mario Paone
