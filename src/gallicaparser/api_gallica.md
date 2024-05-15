# Notes on Gallica API endpoints

## Issues : 

This service allows you to retrieve the years of publication or the list of issues for a given year for a periodical transmitted as a parameter.

Without the date parameter, the service returns the complete list of years in which there is at least one searchable issue, sorted by year.
With a date parameter, the service returns all the issues that can be consulted, sorted by day of publication.This service allows you to retrieve the years of publication or the list of issues for a given year for a periodical transmitted as a parameter.

## OAI:
This service returns the document's OAI record as well as other technical information, such as the type of document, or whether or not full-text searching is available.

Only one parameter is mandatory: the ark of the document's numerical identifier.

## Pagination:

This service returns the pagination of a document.

Only one parameter is mandatory: the ark of the document's numerical identifier.

Pre-calculated image display service:

To retrieve images in broadcast format, we use "image" qualifiers.
    the thumbnail qualifier corresponds to an image with one side measuring 128*192 px
    the lowres qualifier corresponds to an image with one side measuring 256*384 px
    the medres qualifier corresponds to an image with one side measuring 512*768 px
    the highres qualifier corresponds to an image with one side measuring 1024*1536 px

## ContentSearch:

This service returns a list of occurrences of a given word within a document.
The parameters are as follows:

    ark: (mandatory parameter) ARK identifier of the digital document, in the form bpt6kxxxxx or btv1bxxxxx.
    query : (mandatory parameter) search terms to be found in this document
    page : (optional parameter) page number on which to perform the search. This is the number of the <order> tag in the Pagination verb. Use only to retrieve the position on the image where the word is located.
    startResult: (optional parameter) this is used to paginate all the results, bearing in mind that we are limiting the number of elements returned by the service to 10.

## Toc:
This service returns the table of contents.
Only one parameter is mandatory: the ark of the document's numerical identifier.

## texteBrut

By adding ".textBrut" to the end of the url, you get the full text of the document (with bibliographic information in the header)

If you want part of the document, add the qualifier f[X]n[y] at the end of the textBrut qualifier, where X is the number of the page from which you want to obtain the text, and n is the number of subsequent pages.
OCR:

https://gallica.bnf.fr/RequestDigitalElement?O=id&E=ALTO&Deb=x 
	
where :

id is the ark identifier of the digital document,
    x is the number of the page to be extracted

