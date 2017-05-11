---
layout: default
title: Assigment 3
---

ASSIGMENT 3
============

-------------------     ----------------------------

Instrukcie
---------

Pouzitie createpresentation.bat kedy - 1. argument je bud pdf/xhtml, 2.argument je cesta k XSL transformacii, 3.argument je cesta k XML suboru s prezentaciou, ostatne argumenty patria k transformaciam

Poziadavky a ich splnenie
----------
+ Opis dokumentu vytvoreny v XML Schema (schema.xs subor)
	+ Root je presentation element, ten ma lubovolne mnozstvo deti - slidov
	+ Slide samotny moze obsahovat lubovolne mnozstvo obrazkov, ciar, tabuliek ci blokov (textu)
	+ Tabulky obsahuju najprv riadok s hlavickou, nasledne ostatne riadky
	+ Osetrene su aj jednotlive atributy
+ Vytvorenie ukazkovej prezentacie v podobe presentation.xml
+ Vytvorenie XSL transformacii pre XHTML+CSS, presentationXHTML.xsl a presentation.css
+ Vytvorenie PDF transformacii do PDF, presentationPDF.xsl
+ V transformaciach su pouzite for cykly, podmienky, pricom je dany doraz najma na jednotlive templaty - modularnost

Prezentacia do XHTML ma moznost 3 argumentov: *font* - hodnota fontu podla HTML tagu, *background* - background image/color, *heading-font* - farba a font pre heading + table header

Prezentacia do PDF ma moznost 4 argumenty: *font* - hodnota fontu podla XSLFO tagu, *background* - background image/color, *heading-font* - farba a font pre heading + table header, *color* - farba pisma v PDF
  
+ Figures - moznost obrazku s argumentami: width, src, height, align a description
+ Moznost item ako aj ordered zoznamov
+ Vygenerovanie obsahu pomocou elementu toc na lubovolnom slide
+ Automaticke generovanie cislovania stran od 1 az po N
+ Moznost horizontalnej ciary