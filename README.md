# Chi-SquareX-Winter-Internship-Selection-Task
This is the selection task I was given for the winter internship (2024-25) at Chi SquareX

## Prerequisites
* Python 3.10 or higher
* PyPDF2
* Langchain
* Google Generative AI
* Google Cloud SDK

## Installation
1. Clone this repository to your local machine.
2. Create a virtual environment and activate it.
3. Install the required packages by running ```pip install -r My_Requirements.txt```
4. Create a ```.env```file in the root directory of the project and add your Google API key:

```python
GOOGLE_API_KEY=<your_google_api_key>
```
5. Run the application by executing ```python3 MyApp.py file_path "question"```

## Code Structure
* ```DocQA```: The class having all relevant functions
* ```load_pdf```: A function of DocQA class to extract text from the uploaded PDF file and store all pages separately.
* ```create_vectorstore```: A function of DocQA class to split the extracted text into smaller chunks and create a vector database for the text chunks.
* ```setup_qa_chain```: A function of DocQA class to create a conversational chain for question answering.
* ```answer_question```: A function of DocQA class to call all other relevant class functions, give the prompt and generate the result
* ```main```: The main function that uses parser to handle the user input and give output.


## Question Answers
* ### Briefly describe the architecture of your approach
The given PDF document is first read page by page and the text of each page is stored separately along with the page number. Text is segregates pagewise so that the source of the answer can be given. Then, each page is divided into chunks. The 6 most relevant chunks (along with their page numbers) are identified and passed to the LLM as context, along with the user's question. The answer, along with the page numbers of the relevant chunks it given as output.

* ### What are the major pitfalls of your design?
1. The chunks are not intermixed between different pages.
2. Instead of stating the whole page as the source, the exact line/paragraph could have been stated.
3. The number of relevant chunks could have been a variable depending upon the length of the document.

* ### What are some safeguards you would implement if you were to develop a commercial product?
1. End-to-end encryption so that the confidential contracts do not get leaked to the rest of the world.
2. A secure API system.
3. Finetuning the model of legal documents only. This would lead to beter results.
4. Giving no answers instead of giving wrong answers.
