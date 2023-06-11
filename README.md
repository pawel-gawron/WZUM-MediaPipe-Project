## Projekt: Klasyfikacja liter w języku migowym za pomocą biblioteki MediaPipe i klasyfikatorów sklearn oraz TensorFlow

W tym projekcie zaimplementowano system klasyfikacji liter w języku migowym przy użyciu biblioteki MediaPipe oraz różnych klasyfikatorów dostępnych w bibliotece scikit-learn (sklearn) i TensorFlow. Celem projektu było zbudowanie modelu, który na podstawie danych wejściowych, reprezentujących ruchy dłoni w języku migowym, był w stanie poprawnie rozpoznać litery.

### Technologie i biblioteki użyte w projekcie:

- MediaPipe: Biblioteka do przetwarzania multimediów, użyta do śledzenia i analizy ruchów dłoni.
- scikit-learn (sklearn): Biblioteka do uczenia maszynowego, zawierająca różne klasyfikatory.
- TensorFlow: Popularna biblioteka do uczenia maszynowego i tworzenia głębokich sieci neuronowych.
- Optuna: Zewnętrzna biblioteka do optymalizacji hiperparametrów klasyfikatorów.

### Opis działania:

1. **Przygotowanie danych treningowych:** Zbieranie danych obejmowało nagranie sekwencji ruchów dłoni dla każdej litery w języku migowym. MediaPipe została wykorzystana do śledzenia dłoni i pobrania lokalnych informacji dotyczących pozycji palców. Dane treningowe składały się z tych lokalnych informacji.

2. **Ekstrakcja cech:** Na podstawie danych wejściowych dla każdej litery wyodrębniono odpowiednie cechy. Ze względu na wymagania projektu, współrzędne globalne punktów dłoni, współrzędna Z oraz oznaczenie dłoni (lewa/prawa) zostały pominięte w procesie ekstrakcji cech.

3. **Wybór klasyfikatorów:** Do klasyfikacji liter przetestowano różne klasyfikatory dostępne w bibliotece scikit-learn, takie jak SVC, LinearSVC i SGDClassifier. Ponadto, zastosowano również klasyfikator z biblioteki TensorFlow. Klasyfikatory dobrano na podstawie najwyższych wyników ze wszystkich dostępnych klasyfikatorów w bibliotece sklearn aby zawęzić obszar poszukiwań.

4. **Trening i testowanie:** Przystąpiono do treningu klasyfikatorów na danych treningowych, baza zawierała około 4000 wykonanych zdjęć zawartych w pliku excel. Ze względu na ilość danych testowano również podział danych na zbiór treningowy oraz testowy, sprawdzano stosunek od 0.9-0.1 do 0.7-0.3. Każdy klasyfikator został dostosowany do zbioru treningowego, wykorzystując wybrane optymalne hiperparametry. Następnie przetestowano je na danych testowych, aby ocenić ich skuteczność w rozpoznawaniu liter w języku migowym.

5. **Optymalizacja hiperparametrów:** Wykorzystano bibliotekę Optuna do automatycznego doboru optymalnych hiperparametrów dla każdego klasyfikatora. Zadaniem Optuny było znalezienie najlepszych parametrów, które maksymalizują dokładność klasyfikacji na zbiorze walidacyjnym.

6. **Wybór najlepszego klasyfikatora:** Na podstawie wyników testowania wybrano klasyfikator, który osiągnął najlepsze wyniki w klasyfikacji liter w języku migowym. W przypadku tego projektu, klasyfikator SVC okazał się najbardziej skuteczny.

7. **Opracowanie aplikacji:** Na podstawie wybranego klasyfikatora i wytrenowanego modelu opracowano aplikację, która jest w stanie przyjąć dane wejściowe reprezentujące ruchy dłoni w formacie csv i użyć modelu do klasyfikacji liter w języku migowym.

8. **Ocena i dalsze doskonalenie:** System klasyfikacji liter w języku migowym oceniany jest na podstawie wskaźników dokładności klasyfikacji. W zależności od wyników oceny można podjąć działania mające na celu dalsze doskonalenie systemu, takie jak zbieranie większej ilości danych treningowych, optymalizacja parametrów klasyfikatora, ulepszanie procesu ekstrakcji cech lub wypróbowanie innych klasyfikatorów.

9. **Podsumowanie:** Ten projekt ma na celu wykorzystanie biblioteki MediaPipe i różnych klasyfikatorów, takich jak SVC, LinearSVC, SGDC i TensorFlow, do klasyfikacji liter w języku migowym. Poprzez optymalizację hiperparametrów i trening modelu na danych treningowych, projekt ma na celu stworzenie systemu, który może skutecznie rozpoznawać litery na podstawie ruchów dłoni.

