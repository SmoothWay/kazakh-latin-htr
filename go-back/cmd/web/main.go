package main

import (
	"flag"
	"os"
	"sync"

	"github.com/SmoothWay/kazakh-latin-htr/internal/jsonlog"
	"github.com/sugarme/gotch/ts"
)

const (
	gRpcPort = "50001"
)

type Model struct {
	m *ts.CModule
}
type application struct {
	port      int
	modelName string
	logger    *jsonlog.Logger
	wg        sync.WaitGroup
	model     *Model
}

func main() {

	logger := jsonlog.New(os.Stdout, jsonlog.LevelInfo)
	app := application{
		logger: logger,
	}
	flag.StringVar(&app.modelName, "model", "trocr-small-handwritten.pt", "path to model to use")
	flag.IntVar(&app.port, "port", 8080, "API server port")
	model, err := app.LoadModel(app.modelName)
	if err != nil {
		panic(err)
	}
	// go app.gRPCListen()
	app.model = model
	err = app.serve()

	if err != nil {
		logger.PrintFatal(err, nil)
	}
}
