"""Bibliography of the submission report. Cited in the text as <c k="key"/>; only cited entries are printed,
numbered in order of first citation."""

REFS = {
    # heat treatment and physical metallurgy
    "asm4": "ASM International (1991). <i>ASM Handbook, Volume 4: Heat Treating</i>. ASM International, Materials Park, OH.",
    "asm1": "ASM International (1990). <i>ASM Handbook, Volume 1: Properties and Selection: Irons, Steels, and "
            "High-Performance Alloys</i>. ASM International, Materials Park, OH.",
    "krauss2015": "Krauss, G. (2015). <i>Steels: Processing, Structure, and Performance</i>, 2nd ed. ASM International, "
                  "Materials Park, OH.",
    "bhadeshia2017": "Bhadeshia, H.K.D.H. and Honeycombe, R.W.K. (2017). <i>Steels: Microstructure and Properties</i>, "
                     "4th ed. Butterworth-Heinemann, Oxford.",
    "callister2018": "Callister, W.D. and Rethwisch, D.G. (2018). <i>Materials Science and Engineering: An "
                     "Introduction</i>, 10th ed. Wiley, Hoboken, NJ.",
    "grange1961": "Grange, R.A. (1961). Estimating critical ranges in heat treatment of steels. <i>Metal Progress</i>, "
                  "79, 73.",
    "andrews1965": "Andrews, K.W. (1965). Empirical formulae for the calculation of some transformation temperatures. "
                   "<i>Journal of the Iron and Steel Institute</i>, 203, 721–727.",
    "steven1956": "Steven, W. and Haynes, A.G. (1956). The temperature of formation of martensite and bainite in "
                  "low-alloy steels. <i>Journal of the Iron and Steel Institute</i>, 183, 349–359.",
    "km1959": "Koistinen, D.P. and Marburger, R.E. (1959). A general equation prescribing the extent of the "
              "austenite-martensite transformation in pure iron-carbon alloys and plain carbon steels. <i>Acta "
              "Metallurgica</i>, 7, 59–60.",
    "zener1946": "Zener, C. (1946). Kinetics of the decomposition of austenite. <i>Transactions of the AIME</i>, 167, "
                 "550–595.",
    "marder1976": "Marder, A.R. and Bramfitt, B.L. (1976). The effect of morphology on the strength of pearlite. "
                  "<i>Metallurgical Transactions A</i>, 7, 365–372.",
    "hultgren1947": "Hultgren, A. (1947). Isothermal transformation of austenite. <i>Transactions of the ASM</i>, 39, 915.",
    "grossmann1942": "Grossmann, M.A. (1942). Hardenability calculated from chemical composition. <i>Transactions of "
                     "the AIME</i>, 150, 227–259.",
    "grossmann1939": "Grossmann, M.A., Asimow, M. and Urban, S.F. (1939). Hardenability, its relation to quenching, and "
                     "some quantitative data. In <i>Hardenability of Alloy Steels</i>, ASM, Cleveland, 124–196.",
    "astmA255": "ASTM A255. <i>Standard Test Methods for Determining Hardenability of Steel</i>. ASTM International, "
                "West Conshohocken, PA.",
    "astmA370": "ASTM A370. <i>Standard Test Methods and Definitions for Mechanical Testing of Steel Products</i>. ASTM "
                "International, West Conshohocken, PA.",
    "astmE": "ASTM E112 (grain size), E562 (point counting), E140 (hardness conversion) and E8 (tension testing). ASTM "
             "International, West Conshohocken, PA.",
    "iso2566": "ISO 2566-1. <i>Steel: Conversion of Elongation Values, Part 1: Carbon and Low Alloy Steels</i>. "
               "International Organization for Standardization, Geneva.",
    "hj1945": "Hollomon, J.H. and Jaffe, L.D. (1945). Time-temperature relations in tempering steel. <i>Transactions "
              "of the AIME</i>, 162, 223–249.",
    "grange1977": "Grange, R.A., Hribal, C.R. and Porter, L.F. (1977). Hardness of tempered martensite in carbon and "
                  "low-alloy steels. <i>Metallurgical Transactions A</i>, 8, 1775–1785.",
    "watanabe1974": "Watanabe, J., Shindo, Y., Murakami, Y., Adachi, T., Ajiki, S. and Miyano, K. (1974). Temper "
                    "embrittlement of 2¼Cr-1Mo pressure vessel steel. <i>ASME 29th Petroleum Mechanical Engineering "
                    "Conference</i>, Dallas.",
    "en1993": "EN 1993-1-2 (2005). <i>Eurocode 3: Design of Steel Structures, Part 1-2: General Rules, Structural Fire "
              "Design</i>. European Committee for Standardization, Brussels.",
    "kv1984": "Kirkaldy, J.S. and Venugopalan, D. (1984). Prediction of microstructure and hardenability in low alloy "
              "steels. In Marder, A.R. and Goldstein, J.I. (eds.), <i>Phase Transformations in Ferrous Alloys</i>, "
              "TMS-AIME, Warrendale, PA, 125–148.",
    "li1998": "Li, M.V., Niebuhr, D.V., Meekisho, L.L. and Atteridge, D.G. (1998). A computational model for the "
              "prediction of steel hardenability. <i>Metallurgical and Materials Transactions B</i>, 29, 661–672.",
    "collins2023": "Collins, J., Piemonte, M., Taylor, M., Fellowes, J. and Pickering, E. (2023). A rapid, open-source "
                   "CCT predictor for low-alloy steels, and its application to compositionally heterogeneous material. "
                   "<i>Metals</i>, 13, 1168. Data: doi:10.5281/zenodo.7770264.",
    "saunders2003": "Saunders, N., Guo, Z., Li, X., Miodownik, A.P. and Schillé, J.-Ph. (2003). Using JMatPro to model "
                    "materials properties and behavior. <i>JOM</i>, 55(12), 60–65.",
    "trzaska2007": "Trzaska, J. and Dobrzański, L.A. (2007). Modelling of CCT diagrams for engineering and constructional "
                   "steels. <i>Journal of Materials Processing Technology</i>, 192–193, 504–510.",
    "huang2020": "Huang, X. et al. (2020). A combined machine learning model for the prediction of time-temperature-transformation diagrams of "
                 "high-alloy steels. <i>Journal of Alloys and Compounds</i>, 823, 153694.",
    "cao2023": "Cao, Y. et al. (2023). Modeling continuous cooling transformations for HSLA steels with physical "
               "metallurgy guided hereditary machine learning. <i>Metallurgical and Materials Transactions A</i>, 54, "
               "4891–4904.",
    "hedstrom2026": "Hedström, P. et al. (2026). Physics-informed machine learning for steel development: a computational "
                    "framework and CCT diagram modeling. <i>Metallurgical and Materials Transactions A</i>, 57, 2816–2832.",
    "gladman1972": "Gladman, T., McIvor, I.D. and Pickering, F.B. (1972). Some aspects of the structure-property "
                   "relationships in high-carbon ferrite-pearlite steels. <i>Journal of the Iron and Steel Institute</i>, "
                   "210, 916–930.",
    "pickering1978": "Pickering, F.B. (1978). <i>Physical Metallurgy and the Design of Steels</i>. Applied Science "
                     "Publishers, London.",
    "maynier1978": "Maynier, P., Jungmann, B. and Dollet, J. (1978). Creusot-Loire system for the prediction of the "
                   "mechanical properties of low alloy steel products. In Doane, D.V. and Kirkaldy, J.S. (eds.), "
                   "<i>Hardenability Concepts with Applications to Steel</i>, TMS-AIME, Warrendale, PA, 518–545.",
    "hodgson1992": "Hodgson, P.D. and Gibbs, R.K. (1992). A mathematical model to predict the mechanical properties of "
                   "hot rolled C-Mn and microalloyed steels. <i>ISIJ International</i>, 32, 1329–1338.",
    "gladman1997": "Gladman, T. (1997). <i>The Physical Metallurgy of Microalloyed Steels</i>. Institute of Materials, "
                   "London.",
    "gladman1999": "Gladman, T. (1999). Precipitation hardening in metals. <i>Materials Science and Technology</i>, 15, "
                   "30–36.",
    "irvine1967": "Irvine, K.J., Pickering, F.B. and Gladman, T. (1967). Grain-refined C-Mn steels. <i>Journal of the "
                  "Iron and Steel Institute</i>, 205, 161–182.",
    "narita1975": "Narita, K. (1975). Physical chemistry of the groups IVa (Ti, Zr), Va (V, Nb, Ta) and the rare earth "
                  "elements in steel. <i>Transactions of the ISIJ</i>, 15, 145–152.",
    # machine learning and statistics
    "bhadeshia1999": "Bhadeshia, H.K.D.H. (1999). Neural networks in materials science. <i>ISIJ International</i>, 39, "
                     "966–979.",
    "agrawal2014": "Agrawal, A., Deshpande, P.D., Cecen, A., Basavarsu, G.P., Choudhary, A.N. and Kalidindi, S.R. (2014). "
                   "Exploration of data science techniques to predict fatigue strength of steel from composition and "
                   "processing parameters. <i>Integrating Materials and Manufacturing Innovation</i>, 3, 8.",
    "xiong2020": "Xiong, J., Zhang, T.-Y. and Shi, S.-Q. (2020). Machine learning of mechanical properties of steels. "
                 "<i>Science China Technological Sciences</i>, 63, 1247–1255.",
    "shen2019": "Shen, C., Wang, C., Wei, X., Li, Y., van der Zwaag, S. and Xu, W. (2019). Physical metallurgy-guided "
                "machine learning and artificial intelligent design of ultrahigh-strength stainless steel. <i>Acta "
                "Materialia</i>, 179, 201–214.",
    "rasmussen2006": "Rasmussen, C.E. and Williams, C.K.I. (2006). <i>Gaussian Processes for Machine Learning</i>. MIT "
                     "Press, Cambridge, MA.",
    "kennedy2001": "Kennedy, M.C. and O'Hagan, A. (2001). Bayesian calibration of computer models. <i>Journal of the "
                   "Royal Statistical Society B</i>, 63, 425–464.",
    "meredig2018": "Meredig, B. et al. (2018). Can machine learning identify the next high-temperature superconductor? "
                   "Examining extrapolation performance for materials discovery. <i>Molecular Systems Design and "
                   "Engineering</i>, 3, 819–825.",
    "sutton2020": "Sutton, C. et al. (2020). Identifying domains of applicability of machine learning models for "
                  "materials science. <i>Nature Communications</i>, 11, 4428.",
    "lei2018": "Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J. and Wasserman, L. (2018). Distribution-free "
               "predictive inference for regression. <i>Journal of the American Statistical Association</i>, 113, "
               "1094–1111.",
    "lookman2019": "Lookman, T., Balachandran, P.V., Xue, D. and Yuan, R. (2019). Active learning in materials science "
                   "with emphasis on adaptive sampling using uncertainties for targeted design. <i>npj Computational "
                   "Materials</i>, 5, 21.",
    "lundberg2017": "Lundberg, S.M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. "
                    "<i>Advances in Neural Information Processing Systems</i>, 30, 4765–4774.",
    "chen2016": "Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. <i>Proceedings of the 22nd "
                "ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 785–794.",
    "haario2001": "Haario, H., Saksman, E. and Tamminen, J. (2001). An adaptive Metropolis algorithm. <i>Bernoulli</i>, "
                  "7, 223–242.",
    "gelman2013": "Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A. and Rubin, D.B. (2013). <i>Bayesian "
                  "Data Analysis</i>, 3rd ed. CRC Press, Boca Raton, FL.",
    "morris1991": "Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. "
                  "<i>Technometrics</i>, 33, 161–174.",
    "wilcoxon1945": "Wilcoxon, F. (1945). Individual comparisons by ranking methods. <i>Biometrics Bulletin</i>, 1, 80–83.",
    "holm1979": "Holm, S. (1979). A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of "
                "Statistics</i>, 6, 65–70.",
    "pedregosa2011": "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. <i>Journal of Machine "
                     "Learning Research</i>, 12, 2825–2830.",
    "lam2015": "Lam, S.K., Pitrou, A. and Seibert, S. (2015). Numba: a LLVM-based Python JIT compiler. <i>Proceedings of "
               "the Second Workshop on the LLVM Compiler Infrastructure in HPC</i>, 1–6.",
    "abid2019": "Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A. and Zou, J. (2019). Gradio: hassle-free sharing "
                "and testing of ML models in the wild. arXiv:1906.02569.",
    # data
    "birbilis2025": "Birbilis, N., Ghorbani, M. and Zhao, S. (2025). A database of mechanical properties of steels. "
                    "Mendeley Data, V1, doi:10.17632/jmwb9ddd43.1 (CC BY 4.0).",
    "sae": "SAE J403, <i>Chemical Compositions of SAE Carbon Steels</i>, and SAE J1268, <i>Hardenability Bands for "
           "Carbon and Alloy H Steels</i>. SAE International, Warrendale, PA.",
    "metallus2024": "Metallus Inc. (2024). <i>Practical Data for Metallurgists</i>, 19th ed. End-quench hardenability "
                    "bands (pp. 49–59) and combined hardenability charts (p. 68).",
    "smokvina2021": "Smokvina Hanza, S., Marohnić, T., Iljkić, D. and Basan, R. (2021). Artificial neural networks-based "
                    "prediction of hardness of low-alloy steels using specific Jominy distance. <i>Metals</i>, 11, 714 "
                    "(CC BY 4.0).",
    "marohnic2025": "Marohnić, T., Smokvina Hanza, S., Iljkić, D. and Basan, R. (2025). Possibilities of using specific "
                    "Jominy distance in ANN models for predicting low-alloy steels' microstructure. <i>Materials</i>, 18, "
                    "564 (CC BY 4.0).",
    "steelbench": "Volkov, A. et al. (2026). SteelBench v1.0. Zenodo, doi:10.5281/zenodo.18530558.",
    "zenodo_grades": "Carbon and Stainless Steel Grades (2025). Zenodo, doi:10.5281/zenodo.17704410.",
    "carvajal2017": "Carvajal, L., Artigas, A., Monsalve, A. and Arévalo, E. (2017). Monitoring heat treatments in "
                    "steels by a non destructive ultrasonic method. <i>Materials Research</i>, 20 (Suppl. 2), 347–352 "
                    "(CC BY 4.0).",
    "litsources": "The 52 open-access papers of the literature compilation of this work, listed with their DOIs and "
                  "licences in the project repository (data/raw/literature_sources.md).",
    "uhcsdb": "DeCost, B.L., Hecht, M.D., Francis, T., Webler, B.A., Picard, Y.N. and Holm, E.A. (2017). UHCSDB: "
              "UltraHigh Carbon Steel Micrograph DataBase. <i>Integrating Materials and Manufacturing Innovation</i>, "
              "6, 197–205.",
}
