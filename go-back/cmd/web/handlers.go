package main

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/sugarme/gotch/ts"
)

func (app *application) recognize(w http.ResponseWriter, r *http.Request) {
	err := r.ParseMultipartForm(10 << 20) // 10 MB maximum file size
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Extract the file from the request
	file, handler, err := r.FormFile("file")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Check that the uploaded file is an image
	if !strings.HasPrefix(handler.Header.Get("Content-Type"), "image/") {
		http.Error(w, "Uploaded file is not an image", http.StatusBadRequest)
		return
	}

	// Read the file contents into a byte buffer
	buf := bytes.NewBuffer(nil)
	if _, err := io.Copy(buf, file); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Predict
	n := 1
	for i := 0; i < n; i++ {
		input := []float64{0.18, 0.32}
		x := ts.MustOfSlice(input)
		outs, err := app.model.Predict(x)
		if err != nil {
			panic(err)
		}

		for i, out := range outs {
			fmt.Printf("out-%v: %v\n", i, out)
			// x.MustDrop()
		}
		// x.MustDrop()
	}

}

func (app *application) notFound(w http.ResponseWriter, r *http.Request) {
	app.notFoundResponse(w, r)
}

func (app *application) methodNotAllowed(w http.ResponseWriter, r *http.Request) {
	app.methodNotAllowedResponse(w, r)
}
