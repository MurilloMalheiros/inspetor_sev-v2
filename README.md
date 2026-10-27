# Monitor de sonolencia do motorista

Aplicacao em tempo real para o modelo `best.pt`. O YOLO detecta rosto e boca
aberta, enquanto o MediaPipe mede o fechamento real das palpebras. Os dois
modelos rodam de forma assincrona para que a inferencia pesada nao trave a
exibicao da webcam.

- mostrar se o motorista esta normal, sonolento ou dormindo;
- emitir uma sirene continua quando as duas palpebras ficam fechadas por 3 segundos;
- contar um bocejo somente quando a boca fica aberta por tempo suficiente;
- preencher uma barra de cansaco, emitir avisos progressivos e tocar uma
  sirene sem parar ao atingir 100%;
- alertar quando o rosto deixa de aparecer (camera obstruida ou motorista fora
  de posicao).

## Instalacao e uso

No PowerShell, dentro desta pasta:

```powershell
python -m pip install -r requirements.txt
python inspector_motorista.py
```

Na primeira execucao, um modelo facial leve de aproximadamente 3,6 MB sera
baixado. Pressione `Q` ou `Esc` para sair e `R` para zerar a barra depois de uma
pausa; a sirene critica em 100% somente para ao pressionar `R` ou fechar o
programa.
Para usar um video em vez da webcam:

```powershell
python inspector_motorista.py --source caminho\video.mp4
```

Opcoes uteis:

```powershell
python inspector_motorista.py --help
python inspector_motorista.py --confidence 0.40 --sleep-seconds 2.5
```

O audio usa o mixer persistente do Pygame. Para testar o som sem abrir a camera:

```powershell
python inspector_motorista.py --test-sound
```

O padrao considera sono apos 3 segundos continuos com ambas as palpebras fechadas. Cada
bocejo soma 20% a barra e toca uma confirmacao curta. Os limites de aviso progressivo
sao 40%, 65% e 85%; em 100% a sirene fica continua.

Quando o estado muda para `MOTORISTA DORMINDO`, a sirene tambem permanece
continua e para somente depois que os olhos sao reabertos. Os percentuais de
confianca das deteccoes nao sao exibidos; eles continuam sendo usados apenas
internamente pelo modelo.

A tela mostra apenas `Olhos: ABERTOS` ou `Olhos: FECHADOS`. As leituras passam
por suavizacao e confirmacao temporal, evitando que uma falha rapida de apenas
um olho altere o estado. O tempo de olhos fechados tambem aparece na tela. Para
aumentar ainda mais o tempo necessario para o alarme, por exemplo:

```powershell
python inspector_motorista.py --sleep-seconds 4
```

Os sinais dos dois olhos sao combinados. Se um olho estiver parcialmente
oculto ou tiver uma leitura fraca, o outro ainda pode confirmar o fechamento.
As caixas individuais dos olhos do YOLO nao sao mostradas porque nao representam
se o olho esta aberto ou fechado.

O programa solicita 30 FPS da webcam e mantem a interface desacoplada do YOLO.
A taxa efetiva ainda depende da webcam, da CPU e da iluminacao. Para aliviar
ainda mais a CPU, reduza a inferencia do YOLO para 320 pixels:

```powershell
python inspector_motorista.py --yolo-imgsz 320
```

> Este e um sistema assistivo. Ausencia de deteccao nao e uma medicao clinica e
> o programa nao substitui pausas regulares nem praticas seguras de direcao.
