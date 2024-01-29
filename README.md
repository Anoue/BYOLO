# BYOLO
<p dir="auto"><a target="_blank" rel="noopener noreferrer nofollow" href="https://github.com/Anoue/BYOLO/blob/main/AP.png"><img width="100%" src="https://github.com/Anoue/BYOLO/blob/main/AP.png" alt="BYOLO performance plots" style="max-width: 100%;"></a></p>

# Operating environment
<table>
<thead>
<tr>
<th>CUDA</th>
<th>CUDNN</th>
<th>Python</th>
<th>torch</th>
<th>torchvision</th>
</thead>
<tr>
<td>11.3</td>
<td>8.9.2</td>
<td>3.8</td>
<td>2.1.2</td>
<td>0.16.2</td>
</tr>
</table>

# Results at COCO 2017 val
<table>
<thead>
<tr>
<th align="left">Model</th>
<th>Size</th>
<th align="left">mAP<sup>val<br>0.5:0.95</sup></th>
<th align="left">mAP<sup>val<br>0.5</sup></th>
<th>Params<br><sup> (M)</sup></th>
<th>FLOPs<br><sup> (G)</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><a href="https://github.com/Anoue/BYOLO/releases/download/1.0/byolon.pt"><strong>BYOLO-N</strong></a></td>
<td>640</td>
<td align="left">38.3</td>
<td align="left">55.0</td>
<td>1.5</td>
<td>3.8</td>
</tr>
<tr>
<td align="left"><a href="https://github.com/Anoue/BYOLO/releases/download/1.0/byolos.pt"><strong>BYOLO-S</strong></a></td>
<td>640</td>
<td align="left">45.1</td>
<td align="left">62.4</td>
<td>4.7</td>
<td>10.6</td>
</tr>
<tr>
<td align="left"><a href="https://github.com/Anoue/BYOLO/releases/download/1.0/byolom.pt"><strong>BYOLO-M</strong></a></td>
<td>640</td>
<td align="left">50.0</td>
<td align="left">66.7</td>
<td>11.1</td>
<td>24.7</td>
</tr>
<tr>
<td align="left"><a href="https://github.com/Anoue/BYOLO/releases/download/1.0/byolol.pt"><strong>BYOLO-L</strong></a></td>
<td>640</td>
<td align="left">69.0</td>
<td align="left">52.1</td>
<td>20.8</td>
<td>46.7</td>
</tr>
</tbody>
</table>

# Quick Start
<details open="">
<summary> Install</summary>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto"><pre>git clone https://github.com/Anoue/BYOLO.git
<span class="pl-c1">cd</span> BYOLO
pip install -r requirements.txt</pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="git clone https://github.com/Anoue/BYOLO.git
cd BYOLO
pip install -r requirements.txt" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
</details>
