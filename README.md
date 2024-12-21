# RAG_Legal_Epi
Automated Methods to Improve Legal Coding Assessment

# Table of Contents

File | Description
-------|--------
RAG_Automated_Answers_byState.py | Code for Retrieval Augmented Generation (RAG) system to intake a set of questions, and read through a database and provide output answers
RAG_Questions | Question Intake for RAG system
Similarity_Assessment.py | Code for similarity assessment between 2 text files 
CDC_PHLF_Presentation_Raj.pdf | Project Overview 
Heatmap_Data | Folder Containing Results from both human coder similarity assessment and RAG system similarity evalution 


# Project Overview
Background:
The increasing complexity and volume of legal texts in public health policy require
innovative solutions to improve similarity and consistency in legal coding. This study
examines the consistency of manual legal coding approaches and compares them with
computational methods to identify opportunities for automation in legal epidemiology.
The primary aim is to assess the reliability of automated coding techniques against
human-validated coding.

Methods:
The study employs a dual-method approach addressing two research questions: (1) the
consistency of manual coding across human coders, and (2) the alignment between
human and computational coding outcomes. Text data from Public Health Law
Implementation Projects (PHLIP) were processed using Python scripts to evaluate
coder consistency using metrics such as interrater reliability, and cosine similarity. For
computational methods, automated scripts leveraging cosine similarity and retrieval
augmented generation (RAG) models were developed to replicate and validate manual
coding processes. Text masking was applied to mitigate template-based biases.

Results:
Initial findings highlight substantial agreement among human coders for subsets of
questions, particularly binary-coded items, particularly for certain roles. Computational
methods achieved comparable reliability scores, with specific gains in replicability and
scalability. Challenges included managing data privacy, adapting to legal language
nuances, and accounting for data gaps.

Conclusions:
Automated legal coding methods demonstrate potential for enhancing efficiency and
consistency in legal epidemiology. The study underlines the feasibility of applying NLP-
based methods to public health law assessments, paving the way for broader
integration of automated tools in legal and health policy analyses. Future work will refine
these models, expand their application to diverse legal texts, and develop robust
frameworks to assess the relevance and accuracy of legal coding systems.


# Research Aims

How do automated computational methods of legal coding compare to manual legal coding:
What is the observed consistency across all human validated coding approaches ? What areas are consistent, and where are the highest variances observed?
What level of consistency is observed between the answers validated by humans versus computational methods?
  * Compare computational answers to human answer choices: Do both humans and computer choices gravitate towards a common document choice?
  * Are computational approaches better at conducting legal coding for some questions versus others?

# Methods Setup 



# Findings



# References

References:
1. Alice, E., Witt., Anna, Huggins., Guido, Governatori., Joshua, Buckley. Encoding
legislation: a methodology for enhancing technical validation, legal alignment and
interdisciplinarity. Artificial Intelligence and Law, (2023). doi: 10.1007/s10506-
023-09350-1
   
3. Van Rijsbergen, C. (1979, September). Information retrieval: theory and practice.
In Proceedings of the joint IBM/University of Newcastle upon tyne seminar on
data base systems (Vol. 79, pp. 1-14).

5. Leila, Martini., David, Presley., Sarah, Klieger., Scott, Burris. (2015). Scan of
CDC Legal Epidemiology Articles, 2011-2015. Social Science Research Network,
doi: 10.2139/SSRN.2683585



