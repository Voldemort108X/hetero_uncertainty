---

layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Roboto Slab' rel='stylesheet' type='text/css'>

<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>hetero_uncertainty</title>



<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->

<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>
<!-- Global site tag (gtag.js) - Google Analytics -->

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: 'ColfaxAI', 'Helvetica', sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }

  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300; 
  }

IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/leap.png">
<link rel="icon" type="image/png" sizes="32x32" href="/leap.png">
<link rel="icon" type="image/png" sizes="16x16" href="/leap.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/leap.svg" color="#5bbad5">

<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->
<!-- <link rel="shortcut icon" type="image/x-icon" href="leap.ico"> -->
</head>



<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center><h1><strong><br>Heteroscedastic Uncertainty Estimation for Probabilistic Unsupervised Registration of Noisy Medical Images</strong></h1></center>
<center><h2>
    <a href="https://xiaoranzhang.com/">Xiaoran Zhang*</a>&nbsp;&nbsp;&nbsp;
    <a href="">Daniel H. Pak*</a>&nbsp;&nbsp;&nbsp;
    <a href="">Shawn S. Ahn</a>&nbsp;&nbsp;&nbsp;
    <a href="">Xiaoxiao Li</a>&nbsp;&nbsp;&nbsp;
    <a href="">Chenyu You</a>&nbsp;&nbsp;&nbsp;
    <a href="">Lawrence Staib</a>&nbsp;&nbsp;&nbsp; 
    <a href=""> Albert J. Sinusas</a>&nbsp;&nbsp;&nbsp;
    <a href="https://vision.cs.yale.edu/members/alex-wong.html">Alex Wong</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://seas.yale.edu/faculty-research/faculty-directory/james-duncan">James S. Duncan</a>&nbsp;&nbsp;&nbsp;
   </h2>
    <center><h2>
        <a href="https://medicine.yale.edu/bioimaging/ipa/">Yale University</a>&nbsp;&nbsp;&nbsp;
        <a href="https://tea.ece.ubc.ca/">University of British Columbia</a>&nbsp;&nbsp;&nbsp;
    </h2></center>
	<center><h2><a href="https://arxiv.org/abs/2312.00836">Paper</a> | <a href="https://github.com/Voldemort108X/hetero_uncertainty">Code coming soon</a> </h2></center>
<br>



<!-- <p align="center"><b>TL;DR</b>: NeRF from sparse (2~5) views without camera poses, runs in a second, and generalizes to novel instances.</p>
<br> -->

<h1 align="center">Overview</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <a href="./assets/main_framework.png"> <img src="./assets/main_framework.png" style="width:100%;"> </a>
  </td>
      </tr></tbody></table>
<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
  We propose a heteroscedastic uncertainty estimation framework for unsupervised medical image registration. Existing methods rely on objectives (e.g. mean-squared error) that assume a uniform noise level across the image, disregarding the heteroscedastic and input-dependent characteristics of noise distribution in real-world medical images. This further introduces noisy gradients due to undesired penalization on outliers, causing unnatural deformation and performance degradation. To mitigate this, we propose an adaptive weighting scheme with a relative $\gamma$-exponentiated signal-to-noise ratio (SNR) for the displacement estimator after modeling the heteroscedastic noise using a separate variance estimator to prevent the model from being driven away by spurious gradients from error residuals, leading to more accurate displacement estimation. To illustrate the versatility and effectiveness of the proposed method, we tested our framework on two representative registration architectures across three medical image datasets. Our proposed framework consistently outperforms other baselines both quantitatively and qualitatively while also providing accurate and sensible uncertainty measures. Paired t-tests show that our improvements in registration accuracy are statistically significant.
</p></td></tr></table>
</p>
  </div>
</p>

<br>

<hr>
<h1 align="center">Motivation</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <a href="./assets/motivation.png"> <img src="./assets/motivation.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>
<table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Previous unsupervised image registration frameworks utilize an objective, such as mean-squared error, that assumes homoscedastic noise across an image as shown in the left figure. This does not reflect the heteroscedastic and input-dependent characteristics of noise in real-world medical imaging data. To address this issue, we propose a heteroscedastic uncertainty estimation scheme as shown in the right figure to adaptively weigh the data-fidelity term accounting for the non-uniform variations of noise across the image.
</p></td></tr></table>
<br><br>


<hr>


<h1 align="center">Registration accuracy</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
<tbody>
  <tr>
    <td align="center" valign="middle">
      <a href="./assets/vxm_visual-1.png"> <img src="./assets/vxm_visual-1.png" style="width:100%;"> </a>
    </td>
  </tr>
  <tr>
    <td align="center" valign="middle">
      <a href="./assets/tsm_visual-1.png"> <img src="./assets/tsm_visual-1.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>
<table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Qualitative evaluation of the registration accuracy via segmentation warping for all datasets (top two rows: Voxelmorph architecture, bottom two rows: Transmorph architecture). Our method in the last column (overlayed with ground truth ES myocardium segmentation in yellow) predicts more natural and accurate deformations compared to baselines, evidenced by better matching with the ground-truth labels, smoother contour edges, and locally consistent myocardial region.
</p></td></tr></table>
<br>

<hr>
<h1 align="center">Qualitative uncertainty evaluation</h1>
<!-- <h2 align="center">Learned Geometric Knowledge</h2> -->
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <td align="center" valign="middle">
      <a href="./assets/plot_logsigma_I-1.png"> <img
		src="./assets/plot_logsigma_I-1.png" style="width:100%;"> </a>
  </td>
</table>
<table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Qualitative evaluation of the registration accuracy via segmentation warping for all datasets (top two rows: Voxelmorph architecture, bottom two rows: Transmorph architecture. Our method in the last column (overlayed with ground truth ES myocardium segmentation in yellow) predicts more natural and accurate deformations compared to baselines, evidenced by better matching with the ground-truth labels, smoother contour edges, and locally consistent myocardial region.
</p></td></tr></table>
<br>

<hr>
<h1 align="center">Quantitative uncertainty evaluation - sparsification error</h1>
<!-- <h2 align="center">Learned Geometric Knowledge</h2> -->
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <td align="center" valign="middle">
      <a href="./assets/AUSE_ACDC_logsigma_I-1.png"> <img
		src="./assets/AUSE_ACDC_logsigma_I-1.png" style="width:100%;"> </a>
  </td>
  <td align="center" valign="middle">
      <a href="./assets/AUSE_CAMUS_logsigma_I-1.png"> <img
		src="./assets/AUSE_CAMUS_logsigma_I-1.png" style="width:100%;"> </a>
  </td>
</table>
<table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Estimated $\hat{\sigma}_I^2$ and the corresponding weighting map of our proposed framework under Voxelmorph architecture (top row: ACDC; bottom row: CAMUS). Our predictive $\hat{\sigma}_I^2$ qualitatively makes sense since it correlates with the visual correspondence between the fixed image (first column) and the reconstructed image (second column) warped using estimated displacement. Our proposed weighting map also accurately reflects the relative importance between image intensity and predictive variance based on signal-to-noise, leading the displacement estimator to a better performance.
</p></td></tr></table>
<br>


<hr>
<h1 align="center">Incorporating displacement uncertainty</h1>
<!-- <h2 align="center">Learned Geometric Knowledge</h2> -->
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <td align="center" valign="middle">
      <a href="./assets/ACDC_sigmaZ-1.png"> <img
		src="./assets/ACDC_sigmaZ-1.png" style="width:100%;"> </a>
  </td>
</table>
<table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Estimated $\hat{\sigma}_z^2$ of our proposed method framework under Voxelmorph architecture \cite{balakrishnan_voxelmorph_2019} compared with Voxelmorph-diff \cite{dalca_unsupervised_2019}. The second column shows the warped images of both methods using the displacement predicted. We overlay our predicted $\sigma_z^2$ on estimated displacement $z$ in the third column where red indicates higher uncertainty and blue indicates lower. Our estimated $\sigma_z^2$ is able to capture the randomness more accurately in $z$ together with a better registration performance than Voxelmorph-diff when comparing the second column to the fixed image due to our proposed adaptive signal-to-noise weighting strategy.
</p></td></tr></table>
<br>


<hr>
<!-- <table align=center width=800px> <tr> <td> <left> -->
<center><h1>Citation</h1></center>
<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@misc{zhang_heteroscedastic_2023,
	title = {Heteroscedastic {Uncertainty} {Estimation} for {Probabilistic} {Unsupervised} {Registration} of {Noisy} {Medical} {Images}},
	url = {http://arxiv.org/abs/2312.00836},
	publisher = {arXiv},
	author = {Zhang, Xiaoran and Pak, Daniel H. and Ahn, Shawn S. and Li, Xiaoxiao and You, Chenyu and Staib, Lawrence and Sinusas, Albert J. and Wong, Alex and Duncan, James S.},
	year = {2023},
	note = {arXiv:2312.00836 [cs, eess]}
}
</code></pre>
</left></td></tr></table>




<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>

<center><h1>Acknowledgements</h1></center> 
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->

