from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader, JSONLoader

class FileLoader:

    def __init__(self,path="data"):
        self.documents=[]
        self.path=path

    def doc(self):
        self.documents+=DirectoryLoader(self.path,
                                        glob="**/*.pdf",
                                        loader_cls=PyPDFLoader).load()
        
        self.documents+=DirectoryLoader(self.path,
                                        glob="**/*.json",
                                        loader_cls=JSONLoader,
                                        loader_kwargs={"jq_schema": ".",
                                        "text_content": False}).load()        
                                        
        self.documents+=DirectoryLoader(self.path,
                                        glob="**/*.txt",
                                        loader_cls=TextLoader).load()
        
        self.documents+=DirectoryLoader(self.path,
                                        glob="**/*.docx",
                                        loader_cls=Docx2txtLoader).load()
        
        print(f"Total documents loaded:{len(self.documents)}")

        return self.documents