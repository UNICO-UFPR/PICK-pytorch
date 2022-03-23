Drive contendo o modelo treinado com o BID:

https://drive.google.com/drive/folders/1jh8j1rIG0c2dfx84fvYFDZpAp3OafKvu?usp=sharing


No dir data:
- um subdir de anotações para o teste (id.tsv)
- um subdir contendo as imagens de cada anotação (id.jpg)
- um subdir para os resultados (id.txt)


O PICK usa por default todos os arquivos nos dirs enviados pela linha de comando.

Para testar:
python3 test.py test.py --checkpoint <.pth com o modelo salvo> --boxes_transcripts <dir com anotações> --images_path <dir com imagens> --output_folder <dir com resultados> --batch_size 2 --gpu 0

--gpu n pode ser omitido.
