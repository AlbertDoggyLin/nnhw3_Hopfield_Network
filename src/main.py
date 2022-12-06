if __name__=="__main__":
    from UI import App
    from model import readHopfieldData
    app=App(readHopfieldData)
    app.startAppSync()