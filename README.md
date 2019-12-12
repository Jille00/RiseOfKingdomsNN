# Rise Of Kingdoms Neural Nets
Neural net to scan pictures (trained on the game Rise Of Kingdoms)

This is a project done in preperation of an ingame (in the game ROK) event. 
I was too lazy to write down all these statistics and thus made this.

The main directory has two folders: one for pytesseract and one for my own neural net (NN).
The pyesseract folder is used to scan these images with Google's pretrained NN. This only got around 75% accuracy since it is
not trained on this font I guess.
So I made my own NN, you find this in the "own NN" folder. It has three subdirs for three different parts.
1. ReadPower: It is a NN trained to read digits 0-9 and scans the predetermined area and ouputs the digits found.
This is set up so that it will scan the top200 in the game, so it needs screenshots from the top200 leaderboard.
2. ReadNames. This is a NN trained to read all characters (lowercase and uppercase). 
This is set up so that it will scan the top200 players in the game, so it also needs screenshots from the top200 leaderboard.
3. ReadStats. This is s program using both the NN trained in the previous parts. It takes in screenshots of the players stats
and outputs all these stats in a csv file.

TODO
- welp comments are still needed
- in the digits recognision it will often read two 4's next to eachother as one 4. This has nothing to do with the NN but more
with openCV's countour. Still needs a fix ugh

